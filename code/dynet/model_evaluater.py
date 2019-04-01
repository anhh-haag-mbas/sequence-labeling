class ModelEvaluater:
    def __init__(self, model_list):
        self.run_number = 0
        self.model_list = model_list
        self.positive_negative_miss = {name: [] for name in model_list}
        self.negative_positive_miss = {name: [] for name in model_list}
        self.training_times = {name: [] for name in model_list}
        self.training_output = {name: [] for name in model_list}
    
    def new_run(self):
        self.run_number += 1

    def add_training_time(self, model, time): 
        pass

    def add_training_output(self, output):
        pass
