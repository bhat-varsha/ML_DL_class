from sklearn.datasets import load_iris  #iris dataset
from sklearn.ensemble import RandomForestClassifier #learning algorithms
import joblib

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X, y)

# Save the trained model
joblib.dump(model, "iris_random_forest.pkl")

print("Model trained successfully!")
print("Model saved as iris_random_forest.pkl")

"""
ai deployment :
 8 th septemeber:

API : 

we developed api by using flask , express , 
storing the models in the backend ,
load the model somewhere , input data , process data 
object serialization : storage of objects ,pickle and jobbler 
jobler ,third party software , which is aprt of scikit -learn
pickle - python standard library 

both for object serialization means object storage

when to use this 
if we have large numpy arrays , jobblib is best, faster , less memory , and use case scenarios
if we small numpy , pickle  < time optimization can be seen 

dump , load , methods prebuild inside the library  both these nomenclature is present 
dump :
load : never load data in untrusted source 

pickle library for 5000 large dataset(use small dataset)
always use rb = read binary  while loading the data 

.pkl format always in pickle format , and in rb read binary format 

why do we need rb: read binary , to avoid the errors

api in backend : juat create the endpoint
API ENDPOINT ; HOW TO LOAD , PROCESS DATA , MAKE THE DATA READY FOR OUTPUT 

flow :
LOAD THE MODEL , we have preloaded the data using object serialization ,
process the data : ready for model 
run the model inference : model.predict 
validate the results

fastapi , uvicorn , pydantic ,
pip install fastapi uvicorn pydantic joblib


"""