import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")

if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment variables")

supabase: Client = create_client(supabase_url, supabase_key)


class SupabaseDatabase:
    def __init__(self):
        self.client = supabase

    def insert_heart_disease_data(self, data_list, batch_size=100):
        """
        Inserts heart disease data into Supabase in batches to avoid timeouts.
        Includes retry logic and local fallback.
        """
        from time import sleep

        try:
            total = len(data_list)
            print(f"📤 Inserting {total} records to Supabase in batches of {batch_size}...")

            all_results = []
            for i in range(0, total, batch_size):
                batch = data_list[i:i + batch_size]

                for attempt in range(3):  # up to 3 retries
                    try:
                        response = self.client.table('heart_disease_data').insert(batch).execute()
                        if response.data:
                            all_results.extend(response.data)
                        print(f"✅ Inserted batch {i//batch_size + 1} ({len(batch)} records)")
                        break
                    except Exception as e:
                        print(f"⚠️ Attempt {attempt + 1} failed for batch {i//batch_size + 1}: {e}")
                        sleep(2)  # small delay before retry
                else:
                    print(f"❌ Failed to insert batch {i//batch_size + 1} after 3 attempts.")

            print(f"✅ Successfully inserted {len(all_results)} records total.")
            return all_results

        except Exception as e:
            print(f"❌ Supabase insert failed: {e}")
            # ✅ Optional local fallback (cache data)
            fallback_path = "models/fallback_data.csv"
            import pandas as pd, os
            os.makedirs("models", exist_ok=True)
            pd.DataFrame(data_list).to_csv(fallback_path, index=False)
            print(f"💾 Saved data locally to {fallback_path}")
            raise Exception(f"Error inserting data: {str(e)}")


    def get_all_heart_disease_data(self):
        try:
            response = self.client.table('heart_disease_data').select('*').execute()
            return response.data
        except Exception as e:
            raise Exception(f"Error fetching data: {str(e)}")

    def get_data_count(self):
        try:
            response = self.client.table('heart_disease_data').select('id', count='exact').execute()
            return response.count
        except Exception as e:
            raise Exception(f"Error counting data: {str(e)}")

    def save_training_results(self, model_type, model_name, metrics, training_samples):
        """
        Saves model training results to Supabase, converting NumPy types to native Python types.
        """
        import numpy as np

        def to_serializable(obj):
            """Convert numpy datatypes to pure Python datatypes recursively."""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (list, tuple, np.ndarray)):
                return [to_serializable(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: to_serializable(v) for k, v in obj.items()}
            else:
                return obj

        try:
            # ✅ Clean metrics to remove NumPy types
            clean_metrics = to_serializable(metrics)

            data = {
                "model_type": model_type,
                "model_name": model_name,
                "metadata": clean_metrics,
                "training_samples": int(training_samples),
            }

            # ✅ Perform Supabase insert
            response = self.client.table("model_training_results").insert(data).execute()
            print(f"✅ Saved training results for {model_name}")
            return response

        except Exception as e:
            print(f"❌ Error saving training results: {e}")
            raise Exception(f"Error saving training results: {str(e)}")


    def get_training_results(self, model_type=None):
        try:
            query = self.client.table('model_training_results').select('*')
            if model_type:
                query = query.eq('model_type', model_type)
            response = query.order('created_at', desc=True).execute()
            return response.data
        except Exception as e:
            raise Exception(f"Error fetching training results: {str(e)}")

    def save_prediction(self, model_name, input_data, prediction=None, probability=None, cluster=None):
        try:
            data = {
                'model_name': model_name,
                'input_data': input_data,
                'prediction': prediction,
                'probability': probability,
                'cluster': cluster
            }
            response = self.client.table('predictions').insert(data).execute()
            return response.data
        except Exception as e:
            raise Exception(f"Error saving prediction: {str(e)}")

    def get_predictions(self, limit=100):
        try:
            response = self.client.table('predictions').select('*').order('created_at', desc=True).limit(limit).execute()
            return response.data
        except Exception as e:
            raise Exception(f"Error fetching predictions: {str(e)}")

    # def get_statistics(self):
    #     try:
    #         total_data = self.get_data_count()

    #         response = self.client.table('heart_disease_data').select('target').execute()
    #         data = response.data

    #         target_counts = {'0': 0, '1': 0}
    #         for record in data:
    #             target = str(record['target'])
    #             target_counts[target] = target_counts.get(target, 0) + 1

    #         training_results_count = len(self.get_training_results())
    #         predictions_count = len(self.get_predictions(limit=1000))

    #         return {
    #             'total_records': total_data,
    #             'disease_positive': target_counts.get('1', 0),
    #             'disease_negative': target_counts.get('0', 0),
    #             'total_training_runs': training_results_count,
    #             'total_predictions': predictions_count
    #         }
    #     except Exception as e:
    #         raise Exception(f"Error fetching statistics: {str(e)}")
    
    def get_statistics(self):
        """
        Return summary statistics about heart_disease_data.
        Fetches all data in chunks to avoid Supabase 1000-row limit.
        """
        try:
            total_data = self.get_data_count()

            # ✅ Fetch data in chunks (pagination)
            limit = 1000
            offset = 0
            all_targets = []

            while True:
                response = (
                    self.client
                    .table('heart_disease_data')
                    .select('target')
                    .range(offset, offset + limit - 1)
                    .execute()
                )

                batch = response.data or []
                all_targets.extend(batch)

                if len(batch) < limit:
                    break  # no more rows
                offset += limit

            # ✅ Count targets
            target_counts = {'1': 0, '0': 0, 'unknown': 0}

            for record in all_targets:
                target = record.get('target', None)
                if target is None:
                    target_counts['unknown'] += 1
                elif int(target) == 1:
                    target_counts['1'] += 1
                else:
                    target_counts['0'] += 1

            training_results_count = len(self.get_training_results())
            predictions_count = len(self.get_predictions(limit=1000))

            return {
                'total_records': total_data,
                'disease_positive': int(target_counts['1']),
                'disease_negative': int(target_counts['0']),
                'unknown_target': int(target_counts['unknown']),
                'total_training_runs': training_results_count,
                'total_predictions': predictions_count
            }

        except Exception as e:
            raise Exception(f"Error fetching statistics: {str(e)}")

    
    def clear_all_data(self):
        try:
            self.client.table('heart_disease_data').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
            return {'message': 'All data cleared successfully'}
        except Exception as e:
            raise Exception(f"Error clearing data: {str(e)}")
