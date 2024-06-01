from flask import Flask,request,render_template
from pipeline.prediction import CustomData , predicit_pipeline

app = Flask(__name__)

@app.route("/")
def welcome_page():
    return render_template('index.html')

@app.route('/Pred', methods=['GET','POST'])
def pred_page():
    if request.method=='GET':
        return render_template('Pred.html')
    else:
        data = CustomData(ph  = request.form.get('ph'),
        Hardness = request.form.get("Hardness"),
        Solids = request.form.get("Solids"),
        Chloramines = request.form.get("Chloramines"),
        Sulfate  = request.form.get("Sulfate"),
        Conductivity = request.form.get("Conductivity"),
        Organic_carbon = request.form.get("Organic_carbon"),
        Trihalomethanes = request.form.get("Trihalomethanes"),
        Turbidity = request.form.get("Turbidity")
        )
        pred_df = data.get_dataframe()
        print(pred_df)
        pred_pipline = predicit_pipeline()
        results = pred_pipline.predict(pred_df)
        
       
        print(results)
        return render_template('Pred.html',results = results[0])



if __name__=="__main__":
    app.run(host="0.0.0.0",debug =True)   