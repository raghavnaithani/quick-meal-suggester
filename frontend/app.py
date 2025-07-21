import streamlit as st
import requests

# UI Setup
st.title("🍱 Quick Meal Suggestor")
st.markdown("Enter ingredients you have, and get recipe suggestions!")

user_input = st.text_input("Ingredients (comma-separated):", "tomato, cheese, potato")

if st.button("Suggest Recipes"):
    if not user_input.strip():
        st.warning("Please enter ingredients!")
    else:
        ingredients = [x.strip() for x in user_input.split(",")]
        
        with st.spinner("🔍 Finding recipes..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/suggest",
                    json={"ingredients": ingredients}
                ).json()

                if response["status"] == "success":
                    st.success("🎉 Found recipes!")
                    for recipe in response["suggestions"]:
                        st.subheader(recipe["title"])
                        st.markdown(
                            f"**Ingredients:**<br>{'<br>'.join(recipe['NER_cleaned'])}", 
                            unsafe_allow_html=True
                        )
                        st.write("---")  # Divider
                else:
                    st.error(f"Error: {response['message']}")
            except Exception as e:
                st.error(f"🚨 API call failed: {e}")