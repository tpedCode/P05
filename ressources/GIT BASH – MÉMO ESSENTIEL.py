# ======================================================
# GIT BASH – COOKBOOK ESSENTIEL (OPTIMISÉ)
# ======================================================

# ------------------------------------------------------
# CONCEPTS CLÉS (À COMPRENDRE)
# ------------------------------------------------------
# repo local      : code sur ton ordinateur
# repo distant    : code sur GitHub
# staging (index) : zone intermédiaire avant commit
# HEAD            : dernier commit local
# origin/main     : image locale de la branche main sur GitHub
# fetch           : récupérer l’info du distant sans toucher au code
# pull            : fetch + merge

# ------------------------------------------------------
# VÉRIFIER L'ÉTAT GÉNÉRAL
# ------------------------------------------------------
# git --version                             # version de Git
# git status                                # état du repo local
# git log --oneline --graph                 # historique des commits (clair et lisible)

# ------------------------------------------------------
# INITIALISATION / CLONAGE
# ------------------------------------------------------
# git init                                  # créer un repo local
# git clone URL                             # cloner un repo distant

# ------------------------------------------------------
# LIAISON AVEC GITHUB
# ------------------------------------------------------
# git remote -v                             # voir les dépôts distants
# git remote add origin URL                 # lier repo local à GitHub
# git branch -M main                        # renommer la branche principale
# git push -u origin main                   # premier push (liaison définitive)

# ------------------------------------------------------
# WORKFLOW STANDARD (À RESPECTER)
# ------------------------------------------------------
# git status                                # toujours avant toute action
# git add fichier                           # ajouter un fichier précis
# git add .                                 # ajouter tous les changements
# git commit -m "message clair"             # enregistrer un état logique
# git push                                  # envoyer sur GitHub
# git pull                                  # récupérer + fusionner le distant

# # ------------------------------------------------------
# # VOIR LES MODIFICATIONS DISTANTES (SANS RISQUE)
# # ------------------------------------------------------
# git fetch                                 # récupérer l’état du repo distant
# git status                                # Résultats possibles :
                                            # - up to date     → rien à faire
                                            # - behind         → le distant a des commits en plus
                                            # - ahead          → commits locaux non poussés
                                            # - diverged       → historiques différents
# git log HEAD..origin/main --oneline       # commits présents sur GitHub mais absents en local
# git diff HEAD..origin/main                # différences de code avec le distant (pré-visualisation)

# ------------------------------------------------------
# BRANCHES
# ------------------------------------------------------
# git branch                                # lister les branches
# git branch nom_branche                    # créer une branche
# git checkout nom_branche                  # changer de branche
# git checkout -b nom_branche               # créer + changer
# git merge nom_branche                     # fusionner dans la branche courante

# ------------------------------------------------------
# ANNULER / CORRIGER
# ------------------------------------------------------
# git restore --staged fichier              # retirer du staging
# git restore fichier                       # annuler modifs locales
# git reset --soft HEAD~1                   # annuler commit, garder staging
# git reset HEAD~1                          # annuler commit, staging vidé

# ------------------------------------------------------
# CONFLITS
# ------------------------------------------------------
# git pull                                  # modifier les fichiers concernés, peut créer un conflit
# git add fichier
# git commit -m "résolution conflit"

# ------------------------------------------------------
# .GITIGNORE
# ------------------------------------------------------
# Fichier .gitignore (exemples) :
# .env
# __pycache__/
# *.csv

# git rm --cached fichier                   # retirer un fichier déjà suivi

# ------------------------------------------------------
# DIFFÉRENCES & INSPECTION
# ------------------------------------------------------
# git diff                                  # modifs non indexées
# git diff --staged                         # modifs dans le staging
# git show                                  # détail du dernier commit

# ------------------------------------------------------
# BONNES PRATIQUES
# ------------------------------------------------------
# - Toujours vérifier la branche courante
# - Toujours lire git status
# - 1 commit = 1 action logique
# - Ne jamais expérimenter sur main
# - Fetch avant toute analyse du distant

# ------------------------------------------------------
# CHEMINS GIT BASH (WINDOWS)
# ------------------------------------------------------
# C:\Users\Nom\Dossier  →  /c/Users/Nom/Dossier
# Exemple :
# cd /c/Users/barre/Documents/Pro/Reconversion_professionnelle/Formations/Data_Scientist_by_Openclassrooms/P05
