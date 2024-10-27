import datetime
import streamlit as st # type: ignore
import base64
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from v2astronamewithramdomforest import read_business_names, find_consonants

# Load business names
name_df = read_business_names('./resources/Burmese Bussiness Name - Sheet1.csv')

# Remove \n in Name column (this is redundant in this case since it's handled in the read function)
# name_df['Name'] = name_df['Name'].replace('\n','')

# Burmese consonants grouped by days of the week
day_to_consonants = {
    1: ['အ', 'ဥ', 'ဧ'],  # Sunday
    2: ['က', 'ခ', 'ဂ', 'ဃ', 'င'],  # Monday
    3: ['စ', 'ဆ', 'ဇ', 'ဈ', 'ည'],  # Tuesday
    4: ['ယ', 'ရ', 'လ', 'ဝ'],  # Wednesday
    5: ['ပ', 'ဖ', 'ဗ', 'ဘ', 'မ'],  # Thursday
    6: ['ဟ', 'သ'],  # Friday
    7: ['တ', 'ထ', 'ဒ', 'ဓ', 'န'],  # Saturday
}

# Flatten the dictionary for easy lookup of consonant-to-day mapping
consonant_to_day = {consonant: day for day, consonants in day_to_consonants.items() for consonant in consonants}

# Apply the function to each name and create new columns
name_df[['First Consonant', 'Last Consonant']] = name_df['Name'].apply(lambda x: pd.Series(find_consonants(x, consonant_to_day)))

# Reorder columns
name_df = name_df[['First Consonant', 'Last Consonant', 'Name']]

# Shuffle data
shuffle_data = name_df.sample(frac=1)

# Modeling
X = shuffle_data[['First Consonant', 'Last Consonant']]
Y = shuffle_data['Name']

# Initialize and train Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, Y)


st.set_page_config(layout="wide")

# Function to encode the local image to Base64
def get_base64_image(file_path):
    with open(file_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# Path to your local image file
image_path = "./resources/img/bg-1.jpg"
img_base64 = get_base64_image(image_path)
st.markdown(
    f"""
    <style>
   
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
    }}
    p{{
        color: white;        
        }}
    h1{{
        color: white;
        font-size: 28pt;
        }}  
    h2{{    
    font-size: 20pt;
    }}   
    span{{
        font-size: 18pt;
        font-weight:bold;   
        align:center;
        }} 
    .stSelectbox > div {{ width: 70% !important; }}
    
    .stDateInput > div {{ width: 70% !important; }} 

    .streamlit-expanderHeader {{
        font-size: 20px;
        font-weight: bold;
        color: #4CAF50;  /* Change the header color */
    }}
    .streamlit-expanderContent {{
        background-color: #f9f9f9;  /* Change the background color */
        padding: 10px;  /* Add padding */
    }}
    
    </style>
    """,
    unsafe_allow_html=True
)

# Example Streamlit elements
st.markdown("<h1 style='text-align: center; color: white;'>မြန်မာ့ရိုးရာ ဗေဒင် နည်းပညာဖြင့် လုပ်ငန်းအမည်ပေးခြင်း</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: violet;'>AI Baydin ♈ ♉ ♓</h2>", unsafe_allow_html=True)
#st.header("မြန်မာ့ရိုးရာ ဗေဒင် နည်းပညာဖြင့် လုပ်ငန်းအမည်ပေးခြင်း \n :gray[AI Baydin] :aries: :taurus: :pisces:")

#col1, col2, col3  = st.columns(3)    
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown(
    f"""
    <div style='text-align: center;padding-bottom: 100px;'>
        
    </div>
    """, 
    unsafe_allow_html=True
)
    d = st.date_input(
        "မွေးနေ့ ရွေးချယ်ပါ",
        datetime.date(1990, 1, 1),
        min_value=datetime.date(1920, 1, 1),
        max_value=datetime.date(2010, 12, 31)
    )
#st.markdown('<style>.stSelectbox > div { width: 100% !important; }</style>', unsafe_allow_html=True)
# st.write("ရွေးချယ်ထားသည့် မွေးနေ့ - ", d)
    business = st.selectbox(
        "လုပ်ငန်းအမျိုးအစား ရွေးချယ်ပါ",
        ["စားသောက်ကုန်", 
    "ဆေးဝါး", 
    "စက်ပစ္စည်း (ကား၊ ကွန်ပြူတာ ၊ စက်ပစ္စည်း အမျိုးမျိုး )", 
    "လူသုံးကုန်", 
    "အဝတ်အထည်", 
    "အလှကုန်", 
    "လောင်စာဆီ", 
    "ပို့ဆောင်ရေး", 
    "ဆက်သွယ်ရေး", 
    "ဆေးရုံဆေးခန်း", 
    "စားသောက်ဆိုင်", 
    "ဖုန်းဆိုင်", 
    "ဥပဒေ အကြံပေး", 
    "မီးသတ်ပစ္စည်းဆိုင်", 
    "အိမ်ဆောက်ပစ္စည်းဆိုင်", 
    "အလှပြင်ဆိုင်၊ ဆံပင်ညှပ်ဆိုင်", 
    "ပန်း ၊ ပန်းအလှဆင်", 
    "နာရေးပစ္စည်းဆိုင်", 
    "Animal Service", 
    "ဖက်ရှင်ဆိုင်", 
    "နိဗ္ဗန်ကုန်", 
    "အကျိုးဆောင်", 
    "ပွဲရုံလုပ်ငန်း", 
    "ခရီးသွားလုပ်ငန်း", 
    "ပရိဘောဂလုပ်ငန်း", 
    "မိတ္တူ လုပ်ငန်း"
        ]
)
# st.write("ရွေးချယ်ထားသည့် လုပ်ငန်းအမျိုးအစား - ", business)
    city= st.selectbox(
        "မြို့ရွေးချယ်ပါ",
        ("ကျိုက်ထို", 
    "ပေါက်တော", 
    "ကော့မှူး", 
    "ရေဦး", 
    "မင်းကင်း", 
    "လေးမျက်နှာ", 
    "လားရှိုး", 
    "ခေါင်လန်ဖူး", 
    "သရက်", 
    "ဆော့လော်", 
    "ရမည်းသင်း", 
    "စစ်တွေ", 
    "မာန်အောင်", 
    "လင်းခေး", 
    "ကော့သောင်း", 
    "ကွမ်းလုံ", 
    "သာယာဝတီ", 
    "သထုံ", 
    "လမ်းမတော်", 
    "စမ်းချောင်း", 
    "မိုင်းတုံ", 
    "အုတ်ဖို", 
    "လောက်ကိုင်", 
    "ပန်းဘဲတန်း", 
    "စဉ့်ကူး", 
    "ညောင်တုန်း", 
    "သန်လျင်", 
    "မိုင်းဖြတ်", 
    "မြို့သစ်", 
    "တောင်ငူ", 
    "သုံးခွ", 
    "ဝါးခယ်မ", 
    "မြစ်ကြီးနား", 
    "ပေါက်ခေါင်း", 
    "ဆီဆိုင်", 
    "ကျွန်းစု", 
    "ပလက်ဝ", 
    "ဖာပွန်", 
    "မိုးညို", 
    "ထန်းတပင်", 
    "မအူပင်", 
    "တနိုင်း", 
    "မင်းတုန်း", 
    "နားဖန်း", 
    "တောင်ကုတ်", 
    "သာပေါင်း", 
    "မိုင်းရယ်", 
    "ကျောက်ပန်းတောင်း", 
    "ဟားခါး", 
    "ဖားကန့်", 
    "တောင်သာ", 
    "ဆိပ်ဖြူ", 
    "မယ်စဲ", 
    "ပွင့်ဖြူ", 
    "ခရမ်း", 
    "မိုးကောင်း", 
    "မန်တုံ", 
    "မြောင်", 
    "တာချီလိတ်", 
    "ကျိုက်လတ်", 
    "မှော်ဘီ", 
    "ပန်းတောင်း", 
    "ဆားလင်းကြီး", 
    "ဘုတ်ပြင်း", 
    "လဟယ်", 
    "မော်လမြိုင်", 
    "အမရပူရ", 
    "မိုးမိတ်", 
    "မြေပုံ", 
    "ကျောက်ကြီး", 
    "ပင်းတယ", 
    "ချင်းရွှေဟော်", 
    "ဝန်းသို", 
    "နောင်မွန်း", 
    "မိုးနဲ", 
    "မြောက်ဥက္ကလာပ", 
    "မက်မန်း", 
    "ချောင်းဆုံ", 
    "ဓနုဖြူ", 
    "ဖားဆောင်း", 
    "မော်လမြိုင်ကျွန်း", 
    "ရွာငံ", 
    "လှိုင်းဘွဲ့", 
    "တောင်ကြီး", 
    "မုဒုံ", 
    "ရွှေကျင်", 
    "မိုင်းလား", 
    "မံစီ", 
    "ခင်ဦး", 
    "အင်းတော်", 
    "ယောင်လင်း", 
    "ကျောက်မဲ", 
    "မင်္ဂလာဒုံ", 
    "ပုဗ္ဗသီရိ", 
    "ကန်ကြီးထောင့်", 
    "မိုင်းယန်း", 
    "မိုင်းကိုင်", 
    "ဇီးကုန်း", 
    "နမ့်ဆန်", 
    "စစ်ကိုင်း", 
    "ကောလင်း", 
    "နွားထိုးကြီး", 
    "လပွတ္တာ", 
    "တန့်ယန်း", 
    "တိုက်ကြီး", 
    "ပေါက်", 
    "ကမာရွတ်", 
    "ထန်းတပင်", 
    "ဟိုတောင်း", 
    "ကြည့်မြင်တိုင်", 
    "ထီးချိုင့်", 
    "ဒီးမော့ဆို", 
    "လုံထန်", 
    "ထားဝယ်", 
    "သာကေတ", 
    "ဘီးလင်း", 
    "ရွှေကူ", 
    "တာမွေ", 
    "ပန်းတနော်", 
    "မတ္တရာ", 
    "ဝမ်းတွင်း", 
    "မောင်တော", 
    "ပျဉ်းမနား", 
    "ဗန်းမော်", 
    "ပန်ယန်း", 
    "အရာတော်", 
    "စဉ့်ကိုင်", 
    "ဘူးသီးတောင်", 
    "သံတောင်ကြီး", 
    "အိုက်ချန်", 
    "အုတ်တွင်း", 
    "ကလေးဝ", 
    "မန်တွန်း", 
    "တောင်ဥက္ကလာပ", 
    "ဆွမ်ပရာဘွမ်", 
    "ဆောင်ဖ", 
    "သံတွဲ", 
    "ကျောက်တံတား", 
    "သာစည်", 
    "ကံမ", 
    "မိတ္ထီလာ", 
    "ခန္တီး", 
    "သိန္နီ", 
    "ပန်ဆန်း (ပန်ခမ်း)", 
    "မိုးညှင်း", 
    "ဒဂုံ", 
    "ဗန်းမောက်", 
    "ဒဂုံမြို့သစ်", 
    "ပင်လောင်း", 
    "မြဝတီ", 
    "ရေစကြို", 
    "အမ်း", 
    "လသာ", 
    "လှိုင်သာယာ", 
    "ဒလ", 
    "စလင်း", 
    "ကလေး", 
    "လှိုင်", 
    "နန်းယွန်း", 
    "ကန့်ဘလူ", 
    "နတ်မောက်", 
    "မင်းလှ", 
    "ပျော်ဘွယ်", 
    "အင်းစိန်", 
    "ရေတာရှည်", 
    "ကွန်ဟိန်း", 
    "လက်ပံတန်း", 
    "မလှိုင်", 
    "ပခုက္ကူ", 
    "ရသေ့တောင်", 
    "မိုင်းပေါက်", 
    "ရွှေတောင်", 
    "ဘောလခဲ", 
    "မိုင်းခတ်", 
    "ကျေးသီး", 
    "ဒဂုံမြို့သစ်", 
    "သနပ်ပင်", 
    "ကုန်းကြမ်း", 
    "လဲချား", 
    "ကိုကိုးကျွန်း", 
    "ညောင်ဦး", 
    "မြိုင်", 
    "ကောင်မင်ဆန်း", 
    "မြန်အောင်", 
    "ဒေးဒရဲ", 
    "ကလော", 
    "မြင်းခြံ", 
    "မိုင်းဆတ်", 
    "နားကောင်", 
    "အောင်လံ", 
    "ရင်ဖန့်", 
    "မိုင်းကာ", 
    "ဇလွန်", 
    "မောက်မယ်", 
    "လောက်ကိုင်", 
    "နမ်ခမ်းဝူး", 
    "ပဲခူး", 
    "အိမ်မဲ", 
    "မကွေး", 
    "ဖလမ်း", 
    "တန့်ဆည်", 
    "ဗဟန်း", 
    "မဘိမ်း", 
    "ကျွန်းလှ", 
    "မိုင်းပျဉ်း", 
    "ရေကြည်", 
    "ဖရူဆို", 
    "ဇမ္ဗူသီရိ", 
    "ကသာ", 
    "ခွန်းမား", 
    "ကလောင်ဖါ", 
    "ဆင်ပေါင်ဝဲ", 
    "အလုံ", 
    "ထန်တလန်", 
    "နာဝီး", 
    "ကြံခင်း", 
    "နမ့် တစ်", 
    "ကျိုက်မရော", 
    "ဝက်လက်", 
    "မတူပီ", 
    "လွိုင်လင်", 
    "မိုးကုတ်", 
    "ထီးလင်း", 
    "ဒေါပုံ", 
    "မိုင်းပန်", 
    "ကျောင်းကုန်း", 
    "ပြည်ကြီးတံခွန်", 
    "ကျောက်ဖြူ", 
    "နမ္မတူ", 
    "ပုလဲ", 
    "ပြည်", 
    "မူဆယ်", 
    "သင်္ဃန်းကျွန်း", 
    "မိုင်းမော", 
    "ကျောက်တန်း", 
    "သံဖြူဇရပ်", 
    "ကနီ", 
    "ဥတ္တရသီရိ", 
    "လှည်းကူး", 
    "အင်ဂျန်းယန်", 
    "ဘိုကလေး", 
    "သဲကုန်း", 
    "နတ်တလင်း", 
    "နမ့်စန်", 
    "မန်မန်ဆိုင်", 
    "မြစ်သား", 
    "ဖျာပုံ", 
    "တောင်တွင်းကြီး", 
    "ဘားအံ", 
    "ဝေါ", 
    "ဘုတလင်", 
    "ဝိုင်းမော်", 
    "ညောင်ရွှေ", 
    "မိုးမောက်", 
    "မင်းပြား", 
    "ဟိုပုံး", 
    "ဖောင်းပြင်", 
    "ကဝ", 
    "လွိုင်ကော်", 
    "ပန်ဝိုင်", 
    "ဒိုက်ဦး", 
    "ပုသိမ်", 
    "တပ်ကုန်း", 
    "ကြို့ပင်ကောက်", 
    "ဂွ", 
    "ကွမ်း"
    ),
    )

# st.write("ရွေးချယ်ထားသည့်မြို့ - ", city)
# st.write(" - ", d.day)
# st.write(" - ", d.month)
# st.write(" - ", d.year)
    mmyear=d.year-638
    if d.month <= 4:
           if d.month < 4:
              mmyear=mmyear-1
           else:
                if d.day < 12:
                  mmyear=mmyear-1
    birth_number=mmyear%7
    #day_name = d.strftime("%A")
    day_name = d.strftime("%a").lower()

# st.write("Day Name:", day_name)
    st.write("မွေးနှစ် အကြွင်း :", birth_number)
    #st.write("MM year:", mmyear)

# Function to take a single input and get predictions based on how many times it appears in the dataset
def test_with_same_input_duplicate_outputs(start_input, end_input):
    # Convert user input to a DataFrame
    input_data = pd.DataFrame({'First Consonant': [start_input], 'Last Consonant': [end_input]})

    # Count how many times this input appears in the original dataset
    duplicate_count = len(name_df[(name_df['First Consonant'] == start_input) & (name_df['Last Consonant'] == end_input)])

    max_count = min(duplicate_count, 11)

    

    if duplicate_count == 0:
        return "Input not found in the dataset."


    # Combine numerical 'First Consonant' and encoded Last Consonant'
    input_combined = pd.concat([input_data['First Consonant'].reset_index(drop=True),
                                input_data['Last Consonant'].reset_index(drop=True)], axis=1)


    # Get predicted probabilities for each class
    probabilities = clf.predict_proba(input_combined)

    # Randomly select predictions based on probabilities without duplication
    # First, limit the size to the number of available unique classes to avoid over-selecting
    unique_classes = np.unique(clf.classes_)
    selection_size = min(max_count, len(unique_classes))

    # Choose 'selection_size' number of unique predictions
    top_predictions = np.random.choice(unique_classes, size=selection_size, p=probabilities[0], replace=False)

    return top_predictions


import json

# Path to the uploaded JSON file
file_path = './resources/astro.json'

# Open and read the JSON file
with open(file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)  # Parse the JSON data

#remainder dict
remainder = {
    0: "remainder0", 
    1: "remainder1",
    2: "remainder2",  
    3: "remainder3",
    4: "remainder4", 
    5: "remainder5",  
    6: "remainder6",   
}

seven_days = { 
    1: "တနင်္ဂနွေ",
    2: "တနင်္လာ",  
    3: "အင်္ဂါ",
    4: "ဗုဒ္ဓဟူး", 
    5: "ကြာသပတေး",  
    6: "သောကြာ",
    7: "စနေ",
}

r = remainder[birth_number]

# # Access "remainder1" -> "sun"
json_data = data[r][day_name]

# Example: Test the model with the same input and get outputs based on the number of duplicates
start_input = int(json_data['start_num'])  # Example user input for 'Start'
end_input = int(json_data['end_num'])  # Example user input for 'End'
output_name_list = []

# Call the function to predict the outputs based on the number of duplicates
predicted_labels = test_with_same_input_duplicate_outputs(start_input, end_input)

 

bussiness_type = {
        1: "စားသောက်ကုန်",
        2: "ဆေးဝါး",
        3: "စက်ပစ္စည်း (ကား၊ ကွန်ပြူတာ ၊ စက်ပစ္စည်း အမျိုးမျိုး )",
        4: "လူသုံးကုန်",
        5: "အဝတ်အထည်",
        6: "အလှကုန်",
        7: "လောင်စာဆီ",
        8: "ပို့ဆောင်ရေး",
        9: "ဆက်သွယ်ရေး",
        10: "ဆေးရုံဆေးခန်း",
        11: "စားသောက်ဆိုင်",
        12: "ဖုန်းဆိုင်",
        13: "ဥပဒေ အကြံပေး",
        14: "မီးသတ်ပစ္စည်းဆိုင်",
        15: "အိမ်ဆောက်ပစ္စည်းဆိုင်",
        16: "အလှပြင်ဆိုင်၊ ဆံပင်ညှပ်ဆိုင်",
        17: "ပန်း ၊ ပန်းအလှဆင်",
        18: "နာရေးပစ္စည်းဆိုင်",
        19: "Animal Service",
        20: "ဖက်ရှင်ဆိုင်",
        21: "နိဗ္ဗန်ကုန်",
        22: "အကျိုးဆောင်",
        23: "ပွဲရုံလုပ်ငန်း",
        24: "ခရီးသွားလုပ်ငန်း",
        25: "ပရိဘောဂလုပ်ငန်း",
        26: "မိတ္တူ လုပ်ငန်း",
        27: "ပညာရေး",
    }
# Find key(s) corresponding to the target value
b_type_key = [key for key, value in bussiness_type.items() if value == business]
b_type_key_int = int(b_type_key[0])


# Output the predicted labels or error message
if isinstance(predicted_labels, str):
    print(predicted_labels)
else:
    for i, label in enumerate(predicted_labels):
        # Load the Bussiness type data
        file_path = './resources/Astro - B-type.csv'
        b_type_data = pd.read_csv(file_path)
        
        # Find the row index where the value  is in Column
        row_index = b_type_data.loc[b_type_data['Name'] == label].index
        
        # Select value from the second column of a specific row
        value = b_type_data.loc[row_index, 'Invalid'].values[0]
        
        list_values =  [int(x) for x in value.split(',')]
        # Check if  exists in the list
        if b_type_key_int not in list_values:
            output_name_list.append(label)  # Append the value if  does not exist
        
        
        

output_name = ', '.join(map(str, output_name_list))
with col2:
    with st.expander(" ",expanded=True):        
    #st.write("  ")
#with col3:
        st.markdown(
    f"""
    <div style='text-align: center;padding-bottom: 50px;'>
        <span style='color: yellow;font-size:19px;'>တွက်ချက်မှုရလဒ်</span>
    </div>
    """, 
    unsafe_allow_html=True
)
        st.markdown(f"<span style='color:yellow; font-size:17px;'>အဆိုပြု လုပ်ငန်းအမည် :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style='color:white;font-size:15px;'>{output_name}</span>", unsafe_allow_html=True)
        st.markdown(f"""<div style='text-align: center; padding-bottom: 30px'><span style='color:violet; font-size:16px;'>---- {seven_days[int(json_data['start_num'])]}နံ နှင့်စပြီး {seven_days[int(json_data['end_num'])]}နံ နှင့်ဆုံးသော လုပ်ငန်း အမည်ကိုပေးပါ။ ----</span></div>""", unsafe_allow_html=True)
        st.markdown(f"<span style='color:yellow; font-size:17px;'>ကံကောင်းစေသော အရောင် :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style='color:white;font-size:15px;'>{json_data['luck_color']}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:yellow; font-size:17px;'>မင်္ဂလာ အချိန် :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style='color:white;font-size:15px;'>{json_data['luck_time']}</span>", unsafe_allow_html=True)
        st.markdown(f"<span style='color:yellow; font-size:17px;'>ဆောင်ရန်၊ ရှောင်ရန် :&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span><span style='color:white;font-size:15px;'>{json_data['instruction']}</span>", unsafe_allow_html=True)
