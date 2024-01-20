from sklearn.pipeline import Pipeline
from joblib import dump
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt 

def update_model(model: Pipeline) -> None:
    dump(model, 'model/model.pkl')
    
def save_simple_metrics_report(train_score: float, test_score: float, validation_score: float, model: Pipeline) -> None:
    with open('report.txt','w') as report_file:
        report_file.write('# Model Pipeline Description')
        
        for key, value in model.named_steps.items():
            report_file.write(f'### {key}: {value.__repr__()}'+'\n')
            
        report_file.write(f'## Train Score: {train_score}'+'\n')
        report_file.write(f'## Train Score: {test_score}'+'\n')
        report_file.write(f'## Train Score: {validation_score}'+'\n')
        
def get_model_performance_test_set(y_real: pd.Series, y_pred: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(10,10))
    fig.set_figheight(10)
    fig.set_figwidth(10)
    sns.regplot(x=y_real, y=y_pred, ax=ax)
    ax.set_title('Behavior of the model prediction')
    ax.set_xlabel('Real Worldwide Gross')
    ax.set_ylabel('Predicted Worldwide Gross')  
    fig.savefig('prediction_behavior.png')