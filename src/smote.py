
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, ArrayType, FloatType
from pyspark.ml.feature import BucketedRandomProjectionLSH
from pyspark.ml.linalg import Vectors, VectorUDT

class Smote:
    """
    Synthetic Minority Over-sampling Technique (SMOTE) implementation in PySpark.
    Uses LSH for Approximate Nearest Neighbor search to scale to large datasets.
    """
    def __init__(self, 
                 featuresCol="features", 
                 labelCol="label", 
                 bucketLength=2.0, 
                 numHashTables=3, 
                 k=5, 
                 percentage=1.0, 
                 seed=42):
        """
        Args:
            featuresCol (str): Name of the features column (Vector).
            labelCol (str): Name of the label column.
            bucketLength (float): LSH parameter. Control the bucket size.
            numHashTables (int): LSH parameter. Number of hash tables.
            k (int): Number of nearest neighbors to consider.
            percentage (float): Oversampling percentage (1.0 = 100% increase, i.e., double the size).
            seed (int): Random seed.
        """
        self.featuresCol = featuresCol
        self.labelCol = labelCol
        self.bucketLength = bucketLength
        self.numHashTables = numHashTables
        self.k = k
        self.percentage = percentage
        self.seed = seed

    def transform(self, df: DataFrame) -> DataFrame:
        """
        Apply SMOTE to the DataFrame.
        """
        # Identify minority class (assumes binary classification for now or 1 is minority)
        # Ideally, we should calculate counts or let user specify.
        # Here we assume the minority class is the one with fewer counts.
        class_counts = df.groupBy(self.labelCol).count().collect()
        # Sort by count ascending
        class_counts = sorted(class_counts, key=lambda x: x['count'])
        minority_label = class_counts[0][self.labelCol]
        majority_label = class_counts[-1][self.labelCol]
        
        print(f"Detected Minority Class: {minority_label}, Count: {class_counts[0]['count']}")
        print(f"Detected Majority Class: {majority_label}, Count: {class_counts[-1]['count']}")

        minority_df = df.filter(F.col(self.labelCol) == minority_label)
        majority_df = df.filter(F.col(self.labelCol) == majority_label)

        # 1. Find KNN using LSH
        lsh = BucketedRandomProjectionLSH(inputCol=self.featuresCol, 
                                          outputCol="hashes", 
                                          bucketLength=self.bucketLength, 
                                          numHashTables=self.numHashTables, 
                                          seed=self.seed)
        model = lsh.fit(minority_df)

        # Self-join to find pairs (Approximate KNN)
        # Use a loose threshold to ensure we get enough neighbors, or infinite
        print("Searching for neighbors using LSH...")
        pairs = model.approxSimilarityJoin(minority_df, minority_df, threshold=float("inf"), distCol="dist")

        # Filter out self-loops (distance 0 or same ID if we had one, but distance 0 is proxy)
        # Note: LSH may return distance 0 for duplicates.
        # Better to filter strictly distinct rows if we had IDs. 
        # For now, we assume distinct rows have dist > 0 or at least we don't care much if we pick duplicate points.
        pairs = pairs.filter(F.col("dist") > 0)

        # Keep top k neighbors for each point in datasetA
        # We need a unique ID for grouping. If not present, we use monotonic_increasing_id
        # Actually approxSimilarityJoin returns struct cols 'datasetA' and 'datasetB'.
        # We can hashing the vector to group? Unstable.
        # Let's add an ID to minority_df first.
        minority_df = minority_df.withColumn("__id__", F.monotonically_increasing_id())
        
        # We need to re-run approxSimilarityJoin with IDs
        model = lsh.fit(minority_df)
        pairs = model.approxSimilarityJoin(minority_df, minority_df, threshold=float("inf"), distCol="dist")
        pairs = pairs.filter(F.col("datasetA.__id__") != F.col("datasetB.__id__"))

        window = Window.partitionBy("datasetA.__id__").orderBy("dist")
        
        # Get K nearest neighbors
        ranked_pairs = pairs.withColumn("rank", F.row_number().over(window)).filter(F.col("rank") <= self.k)
        
        # 2. Generate Synthetic Samples
        # For 'percentage' amount.
        # If percentage=1.0, we generate 1 new point per minority sample.
        # We pick a random neighbor from the k neighbors.
        
        # Group neighbors into a list
        # We only need the features from datasetB
        neighbors_df = ranked_pairs.groupBy("datasetA.__id__").agg(
            F.collect_list(F.col("datasetB." + self.featuresCol)).alias("neighbors_features")
        )
        
        # Join back with original features
        # datasetA is the origin point.
        base_df = minority_df.join(neighbors_df, on="__id__")
        
        # UDF to generate samples
        @F.udf(returnType=ArrayType(VectorUDT()))
        def generate_samples(base_vec, neighbors_vecs, percentage, seed):
            import random
            random.seed(seed)
            import numpy as np
            
            num_samples = int(percentage)
            remainder = percentage - num_samples
            if random.random() < remainder:
                num_samples += 1
            
            generated = []
            if not neighbors_vecs:
                return []
                
            for _ in range(num_samples):
                neighbor = random.choice(neighbors_vecs)
                # Convert to numpy for easy math
                base_arr = base_vec.toArray()
                neighbor_arr = neighbor.toArray()
                diff = neighbor_arr - base_arr
                gap = random.random()
                new_point = base_arr + gap * diff
                generated.append(Vectors.dense(new_point))
            return generated

        synthetic_rdd = base_df.select(
            F.col(self.featuresCol),
            F.col("neighbors_features"),
            F.lit(self.percentage).alias("percentage"),
            F.lit(self.seed).alias("seed")
        ).rdd.flatMap(lambda row: 
            self._generate_samples_func(row[self.featuresCol], row["neighbors_features"], row["percentage"], row["seed"])
        )
        
        # The above logic with UDF returning array and then explode is cleaner in DataFrame API
        synthetic_df = base_df.withColumn("new_features_array", 
                                          generate_samples(F.col(self.featuresCol), 
                                                           F.col("neighbors_features"), 
                                                           F.lit(self.percentage), 
                                                           F.lit(self.seed)))
        
        synthetic_exploded = synthetic_df.select(F.explode("new_features_array").alias(self.featuresCol))
        
        # Add label column
        synthetic_exploded = synthetic_exploded.withColumn(self.labelCol, F.lit(minority_label))
        
        # Select valid columns (match original schema)
        # We only have features and label now. If original had other cols, they are lost or need defaults.
        # SMOTE usually only works on features.
        
        # Union with original
        final_df = df.unionByName(synthetic_exploded, allowMissingColumns=True)
        
        return final_df

    @staticmethod
    def _generate_samples_func(base_vec, neighbors_vecs, percentage, seed):
        # Helper for RDD if needed, but UDF used above
        pass

