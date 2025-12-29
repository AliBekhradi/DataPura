import os
import joblib
import json

class MappingManager:
    
    def __init__(self,
                 save_path="mappings"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def save_mapping(self,
                     column_name, 
                     mapping, 
                     format: str = "joblib"):
        format = format.lower().strip()
        file_path = os.path.join(self.save_path, f"{column_name}_mapping.{format}")

        if format == "json":
            with open(file_path, "w") as f:
                json.dump(mapping, f)
            print(f"✅ Mapping saved: {file_path}")

        elif format == "joblib":
            file_path = os.path.join(self.save_path, f"{column_name}_mapping.pkl")
            joblib.dump(mapping, file_path)
            print(f"✅ Mapping saved: {file_path}")

        else:
            print("❌ Please enter a valid format ('joblib' or 'json').")

    def load_mapping(self,
                     column_name,
                     format: str = "joblib"):
        format = format.lower().strip()
        file_path = os.path.join(self.save_path, f"{column_name}_mapping.{format}")

        if format == "json":
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    return json.load(f)
            else:
                print(f"❌ Mapping '{column_name}' (JSON) not found in '{self.save_path}'.")

        elif format == "joblib":
            file_path = os.path.join(self.save_path, f"{column_name}_mapping.pkl")
            if os.path.exists(file_path):
                return joblib.load(file_path)
            else:
                print(f"❌ Mapping '{column_name}' (Joblib) not found in '{self.save_path}'.")

        else:
            print("❌ Please enter a valid format ('joblib' or 'json').")

    def apply_mapping(self, series, mapping):
        return series.map(mapping)