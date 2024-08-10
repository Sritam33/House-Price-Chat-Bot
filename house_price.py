from flask import Flask, jsonify, request
from langchain_community.utilities import SQLDatabase
# from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from langchain.prompts import PromptTemplate, FewShotPromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from flask_cors import CORS, cross_origin
# from config import OPENAPI_KEY, DB_HOST, DB_NAME, DB_PASSWORD, DB_PORT, DB_USER


API_KEY = ''
OPENAI_API_KEY = API_KEY
llm = ChatOpenAI(temperature=0.3 , openai_api_key=OPENAI_API_KEY)

db_config = {
    'host': '127.0.0.1',
    'user': 'root',
    'password': 'sritam123#',  # Add your MySQL root password
    'database': 'house_price',
}
host = db_config['host']
port = '3306'
username = db_config['user']
password = db_config['password']
database_schema = db_config['database']
# it is about mysql to python
mysql_uri = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database_schema}"
db = SQLDatabase.from_uri(mysql_uri, include_tables=['sritam'], sample_rows_in_table_info=2)
# connecting llm with our database
db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True, top_k = 2000)

# Initialize the Flask application
app = Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "http://localhost:3000"}})




def retrieve_from_db(query: str) -> str:
    try:
        db_context = db_chain(query)
        print(f"Debug: SQL Query Result - {db_context}")
        
        result = db_context['result']
        print(f"Debug: Extracted Result - {result}")
        
        if isinstance(result, list):
            if len(result) == 1 and isinstance(result[0], tuple):
                result = result[0][0]
            else:
                result = ", ".join(str(r[0]) for r in result)
        else:
            result = str(result)
        
        db_context = result.strip()
        print(f"Debug: Formatted Result - {db_context}")
        return db_context
    except Exception as e: 
        print(f"Error: {e}")
        return "An error occurred while retrieving data from the database."




def generate(query: str, db_context: str) -> str:
    system_message = """
    You are a professional representative of a real estate agency.
    You have to answer user's queries and provide relevant information to help in their house search and price prediction.  
    You have access to everything regarding the house details below.  
    Columns:
    SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
    MSSubClass: The building class
    MSZoning: The general zoning classification
    LotFrontage: Linear feet of street connected to property
    LotArea: Lot size in square feet
    Street: Type of road access
    Alley: Type of alley access
    LotShape: General shape of property
    LandContour: Flatness of the property
    Utilities: Type of utilities available
    LotConfig: Lot configuration
    LandSlope: Slope of property
    Neighborhood: Physical locations within Ames city limits
    Condition1: Proximity to main road or railroad
    Condition2: Proximity to main road or railroad (if a second is present)
    BldgType: Type of dwelling
    HouseStyle: Style of dwelling
    OverallQual: Overall material and finish quality
    OverallCond: Overall condition rating
    YearBuilt: Original construction date
    YearRemodAdd: Remodel date
    RoofStyle: Type of roof
    RoofMatl: Roof material
    Exterior1st: Exterior covering on house
    Exterior2nd: Exterior covering on house (if more than one material)
    MasVnrType: Masonry veneer type
    MasVnrArea: Masonry veneer area in square feet
    ExterQual: Exterior material quality
    ExterCond: Present condition of the material on the exterior
    Foundation: Type of foundation
    BsmtQual: Height of the basement
    BsmtCond: General condition of the basement
    BsmtExposure: Walkout or garden level basement walls
    BsmtFinType1: Quality of basement finished area
    BsmtFinSF1: Type 1 finished square feet
    BsmtFinType2: Quality of second finished area (if present)
    BsmtFinSF2: Type 2 finished square feet
    BsmtUnfSF: Unfinished square feet of basement area
    TotalBsmtSF: Total square feet of basement area
    Heating: Type of heating
    HeatingQC: Heating quality and condition
    CentralAir: Central air conditioning
    Electrical: Electrical system
    1stFlrSF: First Floor square feet
    2ndFlrSF: Second floor square feet
    LowQualFinSF: Low quality finished square feet (all floors)
    GrLivArea: Above grade (ground) living area square feet
    BsmtFullBath: Basement full bathrooms
    BsmtHalfBath: Basement half bathrooms
    FullBath: Full bathrooms above grade
    HalfBath: Half baths above grade
    Bedroom: Number of bedrooms above basement level
    Kitchen: Number of kitchens
    KitchenQual: Kitchen quality
    TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
    Functional: Home functionality rating
    Fireplaces: Number of fireplaces
    FireplaceQu: Fireplace quality
    GarageType: Garage location
    GarageYrBlt: Year garage was built
    GarageFinish: Interior finish of the garage
    GarageCars: Size of garage in car capacity
    GarageArea: Size of garage in square feet
    GarageQual: Garage quality
    GarageCond: Garage condition
    PavedDrive: Paved driveway
    WoodDeckSF: Wood deck area in square feet
    OpenPorchSF: Open porch area in square feet
    EnclosedPorch: Enclosed porch area in square feet
    3SsnPorch: Three season porch area in square feet
    ScreenPorch: Screen porch area in square feet
    PoolArea: Pool area in square feet
    PoolQC: Pool quality
    Fence: Fence quality
    MiscFeature: Miscellaneous feature not covered in other categories
    MiscVal: $Value of miscellaneous feature
    MoSold: Month Sold
    YrSold: Year Sold
    SaleType: Type of sale
    SaleCondition: Condition of sale
    SalePrice:Price of the house 

    

    """
    human_qry_template = HumanMessagePromptTemplate.from_template(
        """
        Input:
        {human_input}

        Context:
        {db_context}

        Output:
        """
    )
    messages = [
        SystemMessage(content=system_message),
        human_qry_template.format(human_input=query, db_context=db_context)
    ]
    try:
        response = llm(messages).content
        print(f"Debug: LLM Response - {response}")
        return response
    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while generating the response."

@app.route('/', methods=['GET'])
def home():
    return "Hello World!"

@app.route('/api/query/', methods=['GET', 'POST'])
@cross_origin()
def get_query_response():
    query_input = request.args.get('q')
    if not query_input:
        return jsonify({"error": "Query parameter 'q' is required."}), 400

    if query_input.lower() in ['hi', 'hello', 'hey']:
        return jsonify({"response": "Hello! How can I assist you to find your dream house?"})

    db_context = retrieve_from_db(query_input)
    response_data = generate(query_input, db_context)
    return jsonify({"response": response_data})


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1',port=5000)