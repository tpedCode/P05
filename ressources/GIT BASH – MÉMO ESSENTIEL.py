# ======================================================
# GIT BASH – MEMO ESSENTIEL
# ======================================================

# --------- VÉRIFIER L'ENVIRONNEMENT ---------
# git --version                 # version Git installée
# git status                    # état du dépôt
# git log --oneline             # historique condensé des commits

# --------- INIT / CLONER ---------
# git init                      # init dépôt local
# git clone URL                 # cloner repo distant

# --------- LIAISON AVEC GITHUB ---------
# git remote -v                 # liste dépôts distants
# git remote add origin URL     # lier repo local à GitHub
# git branch -M main            # renommer branche principale
# git push -u origin main       # premier push

# --------- WORKFLOW DE BASE ---------
# git status                    # toujours vérifier avant d’agir
# git add fichier               # ajouter un fichier précis
# git add .                     # ajouter tous les fichiers modifiés + nouveaux
# git commit -m "msg"           # commit ce qui est dans l’index
# git commit -a -m "msg"        # commit tous les fichiers suivis modifiés
# git push                      # envoyer sur GitHub
# git pull                      # récupérer et fusionner changements distants

# --------- BRANCHES ---------
# git branch                    # liste branches (* = courante)
# git branch nom                # créer une branche
# git checkout nom              # changer de branche
# git checkout -b nom           # créer + changer de branche
# git merge nom                 # fusionner branche dans branche courante

# --------- CORRECTION D'ERREURS ---------
# git restore --staged fichier  # annuler git add
# git restore fichier           # annuler modifications locales
# git reset --soft HEAD~1       # annuler dernier commit, garder staging
# git reset HEAD~1              # annuler dernier commit, fichiers non indexés

# --------- CONFLITS ---------
# git pull                      # peut créer conflit
# modifier le fichier pour résoudre
# git add fichier
# git commit -m "résolution conflit"

# --------- .gitignore ---------
# Créer fichier .gitignore
# Exemples : .env, __pycache__/, *.csv
# git rm --cached fichier       # retirer un fichier déjà suivi

# --------- DIFFÉRENCES ---------
# git diff                      # différences avec dernier commit
# git diff --staged             # différences indexées
# git show                      # détail dernier commit

# --------- BONNES PRATIQUES ---------
# Toujours vérifier branche : git branch
# Toujours vérifier état : git status
# Ajouter tout avant commit : git add .
# Commit clair, 1 action logique = 1 commit
# Ne pas tester directement sur main

# --------- CHEMINS GIT BASH ---------
# Windows C:\… → /c/…
# Exemple :
# cd /c/Users/barre/Documents/Pro/.../P05
