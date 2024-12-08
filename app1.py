import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger le fichier CSV
csv_path = r"C:\Users\OTMANE\Downloads\PROJET 2\df_movie_tmdb.csv"

try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    st.error(f"Le fichier {csv_path} est introuvable. Assurez-vous qu'il existe et que le chemin est correct.")
    st.stop()

# Vérification des colonnes nécessaires
required_columns = ['Titre', 'Genres', 'Année', 'Acteur_1', 'Acteur_2', 'Acteur_3', 'Acteur_4', 'Acteur_5', 'URL_AFFICHE']
if not all(col in df.columns for col in required_columns):
    st.error(f"Le fichier CSV doit contenir les colonnes suivantes : {', '.join(required_columns)}")
    st.stop()

# Renommer les colonnes
df.rename(columns={'Titre': 'Title', 'Genres': 'Genres', 'Année': 'Year'}, inplace=True)

# Combiner les colonnes d'acteurs en une seule colonne
actor_columns = ['Acteur_1', 'Acteur_2', 'Acteur_3', 'Acteur_4', 'Acteur_5']
df['All_Actors'] = df[actor_columns].fillna('').agg(', '.join, axis=1)

# Nettoyer les colonnes
df['Title'] = df['Title'].str.strip().str.lower()
df['Genres'] = df['Genres'].fillna('')
df['All_Actors'] = df['All_Actors'].str.strip()

# Préparation du TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Title'])

# Fonction pour trouver les titres similaires
def find_similar_titles(input_text, tfidf_matrix, df, top_n=3):
    input_vector = tfidf_vectorizer.transform([input_text.lower()])
    similarity_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
    similar_indices = similarity_scores.argsort()[-top_n:][::-1]
    similar_titles = df.iloc[similar_indices]
    return similar_titles[['Title', 'Year', 'Genres', 'All_Actors', 'URL_AFFICHE']]

# Interface Streamlit
st.title("🎥 Recommender System de Films")
st.subheader("Trouvez votre prochain film préféré!")

# Filtres dans la sidebar
with st.sidebar:
    st.header("Filtres")
    # Filtre par année
    year_filter = st.slider("Choisir une plage d'années", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()),
                            value=(int(df['Year'].min()), int(df['Year'].max())))
    
    # Filtre par genres
    genre_filter = st.selectbox("Choisir un genre", ['Tout'] + df['Genres'].dropna().unique().tolist())

# Application des filtres
filtered_df = df
if genre_filter and genre_filter != 'Tout':
    filtered_df = filtered_df[filtered_df['Genres'].str.contains(genre_filter, na=False)]
if year_filter:
    filtered_df = filtered_df[(filtered_df['Year'] >= year_filter[0]) & (filtered_df['Year'] <= year_filter[1])]

# Fonctionnalité principale - Choisir un film
st.write("## Trouver des films similaires")
movie_input = st.selectbox("Choisissez un film :", filtered_df['Title'].tolist())

if movie_input:
    # Afficher le film choisi
    selected_movie = filtered_df[filtered_df['Title'] == movie_input.lower()].iloc[0]
    st.write(f"### **Film choisi : {selected_movie['Title'].capitalize()}** ({int(selected_movie['Year'])})")
    
    # Affiche du film choisi
    if pd.notna(selected_movie['URL_AFFICHE']) and selected_movie['URL_AFFICHE'].startswith('http'):
        st.image(selected_movie['URL_AFFICHE'], caption=selected_movie['Title'], use_column_width=True)
    else:
        st.warning(f"Affiche indisponible pour **{selected_movie['Title']}**")
    
    st.write(f"**Genres**: {selected_movie['Genres']}")
    st.write(f"**Acteurs**: {selected_movie['All_Actors']}")
    st.write("---")
    
    # Trouver les suggestions basées sur les genres
    st.write(f"### Suggestions similaires à **{selected_movie['Title'].capitalize()}** :")
    genre_movies = filtered_df[filtered_df['Genres'].str.contains(selected_movie['Genres'], na=False)].head(3)

    if not genre_movies.empty:
        for index, row in genre_movies.iterrows():
            st.write(f"### **{row['Title'].capitalize()}** ({int(row['Year'])})")
            
            # Afficher l'image du film
            if pd.notna(row['URL_AFFICHE']) and row['URL_AFFICHE'].startswith('http'):
                st.image(row['URL_AFFICHE'], caption=row['Title'], use_column_width=True)
            else:
                st.warning(f"Affiche indisponible pour **{row['Title']}**")
            
            st.write(f"**Genres**: {row['Genres']}")
            st.write(f"**Acteurs**: {row['All_Actors']}")
            st.write("---")
    else:
        st.warning("Aucun film similaire trouvé.")

# Fonctionnalité secondaire - Films par acteur
st.write("## Trouver des films par acteur")
actor_filter = st.selectbox("Choisissez un acteur :", ['Tous'] + sorted(set(', '.join(df['All_Actors']).split(', '))))

if actor_filter and actor_filter != 'Tous':
    st.write(f"### Films avec **{actor_filter}**:")
    actor_movies = df[df['All_Actors'].str.contains(actor_filter, na=False, regex=False)]

    if not actor_movies.empty:
        for index, row in actor_movies.iterrows():
            st.write(f"### **{row['Title'].capitalize()}** ({int(row['Year'])})")
            
            # Afficher l'image du film
            if pd.notna(row['URL_AFFICHE']) and row['URL_AFFICHE'].startswith('http'):
                st.image(row['URL_AFFICHE'], caption=row['Title'], use_column_width=True)
            else:
                st.warning(f"Affiche indisponible pour **{row['Title']}**")
            
            st.write(f"**Genres**: {row['Genres']}")
            st.write(f"**Acteurs**: {row['All_Actors']}")
            st.write("---")
    else:
        st.warning(f"Aucun film trouvé avec **{actor_filter}**.")
















