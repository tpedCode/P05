import os
import pandas as pd

# Dossier contenant tes CSV
folder_path = r"C:\Users\barre\Documents\Pro\Reconversion professionnelle\Formations\Data Scientist by Openclassrooms\P05\data"

# Nom du fichier SQL à générer
output_sql_file = r"C:\Users\barre\Documents\Pro\Reconversion professionnelle\Formations\Data Scientist by Openclassrooms\P05\ressources\generate_import_sql.sql"

# Fonction pour deviner le type SQL
def sql_type(series):
    if pd.api.types.is_integer_dtype(series):
        return "INT"
    elif pd.api.types.is_float_dtype(series):
        return "FLOAT"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "DATE"
    else:
        return "VARCHAR(255)"

# Stocker tout le SQL
all_sql = []

# Boucle sur tous les CSV
for filename in os.listdir(folder_path):
    if filename.endswith(".csv"):
        filepath = os.path.join(folder_path, filename)
        df = pd.read_csv(filepath)
        table_name = os.path.splitext(filename)[0]

        # Générer CREATE TABLE
        columns_sql = []
        for i, col in enumerate(df.columns):
            col_type = sql_type(df[col])
            if i == 0 and col_type == "INT":  # première colonne comme clé primaire si INT
                columns_sql.append(f"{col} {col_type} PRIMARY KEY")
            else:
                columns_sql.append(f"{col} {col_type}")
        create_table_sql = f"CREATE TABLE {table_name} (\n  " + ",\n  ".join(columns_sql) + "\n);"

        # Générer LOAD DATA
        load_data_sql = f"""LOAD DATA LOCAL INFILE '{filepath.replace("\\","\\\\")}'
INTO TABLE {table_name}
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\\n'
IGNORE 1 ROWS;"""

        # Ajouter au SQL total
        all_sql.append(f"-- Table {table_name}\n{create_table_sql}\n{load_data_sql}\n")

# Écrire tout dans un fichier SQL
with open(output_sql_file, "w", encoding="utf-8") as f:
    f.write("\n\n".join(all_sql))

print(f"Fichier SQL généré : {output_sql_file}")
