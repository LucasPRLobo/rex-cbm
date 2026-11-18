from scripts.run_cbm import load_test_records, CBM, CHECKPOINTS
from scripts.concept_experiment import explain_all_active                      
import torch 


if __name__ == "__main__":
    records = load_test_records()
    models = list(CHECKPOINTS.keys())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cbm = CBM("joint",device)
    model_results = {}
    for record in records:
        id_ = record["id"]
        model_results[id_] = explain_all_active(cbm,record,device)
        break
        
