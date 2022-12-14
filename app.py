import pickle
from django.shortcuts import render
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
import json

app=Flask(__name__)
## Load the model

# with open('regmodelOHE.pkl', 'rb') as f:
#     regmodel = pickle.load(f)

# with open('scalerOHEF.pkl', 'rb') as k:
#     scalar = pickle.load(k)


with open('LgtRModel1.pkl', 'rb') as f:
    regmodel = pickle.load(f)

with open('scalerLEC1.pkl', 'rb') as k:
    scalar = pickle.load(k)

@app.route('/')
def home():
    return render_template('home3.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data= request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1,-1))
    output=regmodel.predict(new_data)
    print(output[0])
    return json.dumps(output[0], default=str)
    # return jsonify(output[0])

@app.route('/predict', methods=['POST']) 
def predict():
    data = [float(x) for x in request.form.values()]
    MONTANTE = data[0]
    TAXA_JUROS = data[1]/100
    PRAZOC = data[2]
    TIPOCREDITO = data[3]
    TIPOCLIENTE = data[4]
    DESPESAS = data[5]

    ANOSIDADE = data[6]
    GENERO = data[7]
    ESTADOCIVIL = data[8]
    PAIS = data[9]
    PROVINCIA = data[10]
    TRABALHO = data[11]
    CATEGORIA = data[12]
    SALARIO = data[13]
    ESCOLA = data[14]
    LICENCIADO = data[15]
    
    NR_PRESTACOES                  = PRAZOC
    NR_PREST_PAGAS_ATRASO          = 0
    NR_PREST_PAGAS_SEM_ATRASO      = 0
    NR_PREST_NAOPAGAS_ATRASO       = 0
    NR_PREST_NAO_PAGAS_SEM_ATRASO  = PRAZOC
    PRAZO                          = PRAZOC

    CAPACIDADEENDIVIDAMENTO = 0
    if (0.3*(SALARIO-DESPESAS)) > SALARIO :
        CAPACIDADEENDIVIDAMENTO = 0
    else:
        CAPACIDADEENDIVIDAMENTO = 1

    VALORCREDITO                = MONTANTE

    CAPITAL = MONTANTE/data[2]
    VALORJUROS = 0
    if TIPOCREDITO == 1 :
        VALORJUROS = (TAXA_JUROS/12)*MONTANTE
    else:
        VALORJUROS = TAXA_JUROS*MONTANTE
    
    VALORPRESTACAO                 = CAPITAL + VALORJUROS

    IDADE                          = ANOSIDADE
    PROFISSAO                      = TRABALHO
    RENDAMENSAL                    = SALARIO
    RENDAANUAL                     = SALARIO*12
    HABILITACOES                   = ESCOLA
    SEXO                           = GENERO
    GRADUADO                       = LICENCIADO
    TIPOCREDITO_1                  = TIPOCREDITO
    PAIS_1                         = PAIS
    PROVINCIA_1                    = PROVINCIA

    CATEGORIAPROFISSIONAL_1        = CATEGORIA
    TIPOCLIENTE_1                  = TIPOCREDITO
    ESTADOCIVIL_1 = ESTADOCIVIL
    
    dados = [NR_PRESTACOES, NR_PREST_PAGAS_ATRASO, NR_PREST_PAGAS_SEM_ATRASO,
             NR_PREST_NAOPAGAS_ATRASO, NR_PREST_NAO_PAGAS_SEM_ATRASO, PRAZO,
             CAPACIDADEENDIVIDAMENTO, VALORCREDITO, VALORPRESTACAO, IDADE,
             PROFISSAO, RENDAMENSAL, RENDAANUAL, HABILITACOES, SEXO,
             GRADUADO, TIPOCREDITO_1, PAIS_1, PROVINCIA_1,
             CATEGORIAPROFISSIONAL_1, TIPOCLIENTE_1, ESTADOCIVIL_1]


    final_input=scalar.transform(np.array(dados).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home3.html",prediction_text=" {} ".format(output))

# arr = {
#     "NR_PRESTACOES" = 5, 
# "NR_PREST_PAGAS_ATRASO" = 2, 
# "NR_PREST_PAGAS_SEM_ATRASO" = 3,
# "NR_PREST_NAOPAGAS_ATRASO" = 5, 
# "NR_PREST_NAO_PAGAS_SEM_ATRASO" = 7,
# "TIPOCREDITO" = 1,
# }

# def predict():
#     # data = [float(x) for x in request.form.values()]
#     data = request.form.values()
#     # final_input=scalar.transform(np.array(data).reshape(1,-1))
#     # print(final_input)
#     # output = regmodel.predict(final_input)[0]
#     return render_template("home3.html",prediction_text=data)

if __name__=="__main__":
    app.run(debug=True)