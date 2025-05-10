# Hunyuan3D Glasses Adaptation

Ce projet adapte le modèle Hunyuan3D pour reconstruire des lunettes 3D à partir d'images 2D. Il permet de générer des modèles 3D de lunettes à partir de simples images 2D, avec des matériaux réalistes et des fonctionnalités d'essayage virtuel.

## Contenu
- `src/` : Scripts principaux
- `data/` : Données d'entraînement et de test
- `examples/` : Images d'exemple pour tester le modèle
- `checkpoints/` : Sauvegarde du modèle
- `results/` : Visualisations et résultats générés
- `kaggle_endpoint.py` : Script pour utiliser le projet sur Kaggle
- `kaggle_setup.py` : Script d'installation pour Kaggle
- `run_glasses_pipeline.py` : Script principal pour exécuter le pipeline complet

## Installation simplifiée

La méthode la plus simple pour installer et utiliser ce projet est d'utiliser le script `run_glasses_pipeline.py` :

```bash
# Cloner le dépôt
git clone https://github.com/amin8452/Pfa.git
cd Pfa

# Installer les dépendances et télécharger le modèle
python run_glasses_pipeline.py --mode setup

# Collecter des données d'images de lunettes
python run_glasses_pipeline.py --mode data

# Lancer l'interface web
python run_glasses_pipeline.py --mode web
```

## Installation manuelle

Si vous préférez une installation manuelle :

1. Installez les dépendances :
```bash
pip install -r requirements.txt
```

2. Téléchargez le modèle Hunyuan3D-2 :
```bash
python kaggle_setup.py --download_hunyuan
```

3. Créez les répertoires nécessaires :
```bash
python kaggle_setup.py --create_dirs
```

## Préparation des données

1. Créez un répertoire pour les données de lunettes :
```bash
mkdir -p data/glasses/{train,val,test}
```

2. Organisez vos données selon la structure suivante :
```
data/glasses/
├── train/
│   ├── images/
│   └── meshes/
├── val/
│   ├── images/
│   └── meshes/
├── test/
│   ├── images/
│   └── meshes/
├── train_metadata.json
├── val_metadata.json
└── test_metadata.json
```

3. Les fichiers metadata.json doivent avoir le format suivant :
```json
[
  {
    "id": "glasses_001",
    "image_path": "train/images/glasses_001.jpg",
    "mesh_path": "train/meshes/glasses_001.obj",
    "texture_path": "train/textures/glasses_001.png"
  },
  ...
]
```

## Utilisation

### Génération de lunettes 3D à partir d'une image

Utilisez le script `glasses_demo.py` pour générer un modèle 3D de lunettes à partir d'une image :

```bash
python src/glasses_demo.py --image path/to/glasses_image.jpg --output_dir results
```

Pour utiliser le modèle Hunyuan3D original comme base :

```bash
python src/glasses_demo.py --image path/to/glasses_image.jpg --output_dir results --use_hunyuan
```

### Entraînement du modèle

Pour entraîner le modèle simple :

```bash
python src/train_glasses.py --data_root data/glasses --output_dir checkpoints
```

Pour entraîner le modèle adapté de Hunyuan3D :

```bash
python src/train_glasses.py --data_root data/glasses --output_dir checkpoints --use_hunyuan
```

Pour un entraînement plus avancé avec plus d'options :

```bash
python src/finetune_glasses.py --data_root data/glasses --output_dir checkpoints --checkpoint tencent/Hunyuan3D-2
```

### Évaluation du modèle

Pour évaluer le modèle sur un ensemble de test :

```bash
python src/evaluate_glasses.py --checkpoint checkpoints/best_model.pth --data_root data/glasses --output_dir results
```

## Architecture du modèle

Notre modèle `GlassesHunyuan3DModel` adapte l'architecture Hunyuan3D pour la reconstruction 3D de lunettes. Il comprend :

1. Un encodeur d'image basé sur CLIP pour extraire des caractéristiques robustes
2. Des couches spécifiques aux lunettes pour capturer les détails particuliers des montures et des verres
3. Un décodeur pour générer les vertices 3D
4. Un décodeur pour générer les faces (triangles)
5. Un générateur de texture pour créer des textures réalistes

## Métriques d'évaluation

Nous utilisons plusieurs métriques pour évaluer la qualité de la reconstruction 3D :

- **Chamfer Distance** : Mesure la distance entre les points du modèle prédit et ceux du modèle de référence
- **F-score** : Évalue la précision et le rappel de la reconstruction géométrique
- **Cohérence des normales** : Vérifie si les normales des surfaces sont correctement orientées
- **Similarité de texture** : Compare la qualité des textures générées avec les textures de référence

## Fonctionnalités avancées

### Modèle amélioré
Nous avons amélioré le modèle de base avec des fonctionnalités spécifiques pour les lunettes :
- Architecture d'attention pour se concentrer sur les caractéristiques importantes
- Séparation des montures et des verres pour un traitement spécialisé
- Prédiction des propriétés matérielles pour un rendu réaliste

### Propriétés matérielles réalistes
Notre système prend en charge des propriétés matérielles avancées :
- Métallique : contrôle l'aspect métallique des montures
- Rugosité : affecte la réflexion de la lumière
- Transparence : pour les verres transparents ou teintés
- Indice de réfraction : pour des effets optiques réalistes
- Spéculaire : pour les reflets brillants
- Clearcoat : pour les finitions vernies
- Anisotropie : pour les effets directionnels

### Essayage virtuel
Vous pouvez essayer virtuellement les lunettes générées sur une photo de visage :
```bash
python src/virtual_tryon.py --glasses results/glasses.glb --face path/to/face.jpg --output results/tryon.jpg
```

### Personnalisation des lunettes
Notre outil de personnalisation vous permet de modifier les lunettes générées :
```bash
python src/glasses_customizer.py --output_dir results
```

Fonctionnalités de personnalisation :
- Modification des couleurs des montures et des verres
- Ajustement des dimensions (largeur, hauteur, profondeur)
- Personnalisation des propriétés matérielles
- Prévisualisation sur un visage

### Interface Web
Nous proposons deux interfaces web pour faciliter l'utilisation :
- Interface de génération : pour créer des modèles 3D à partir d'images
- Interface de personnalisation : pour modifier les modèles générés

Pour lancer les deux interfaces :
```bash
python glasses_app.py
```

## Utilisation sur Kaggle

Ce projet est conçu pour fonctionner facilement sur Kaggle. Voici comment l'utiliser :

1. Créez un nouveau notebook Kaggle avec accélérateur GPU (T4 recommandé)

2. Copiez et collez le code suivant :

```python
# Cloner le dépôt GitHub
!git clone https://github.com/amin8452/Pfa.git
%cd Pfa

# Configurer l'environnement
!python kaggle_setup.py --all

# Afficher les exemples d'images disponibles
import matplotlib.pyplot as plt
from PIL import Image
import os

example_dir = "examples"
examples = [f for f in os.listdir(example_dir) if f.endswith(('.jpg', '.png'))]

plt.figure(figsize=(15, 10))
for i, example in enumerate(examples):
    img = Image.open(os.path.join(example_dir, example))
    plt.subplot(2, 2, i+1)
    plt.imshow(img)
    plt.title(example)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Lancer l'endpoint Kaggle
!python kaggle_endpoint.py --use_hunyuan
```

3. Exécutez le notebook
4. Utilisez le lien public généré pour accéder à l'interface web
5. Téléchargez une image de lunettes ou utilisez les exemples fournis
6. Générez votre modèle 3D et téléchargez-le

## Limitations et travaux futurs

- Amélioration de la précision pour les lunettes avec des formes complexes
- Support pour plus de types de lunettes (solaires, correctrices, etc.)
- Génération de matériaux encore plus réalistes (réflexions, transparence, etc.)
- Amélioration de l'essayage virtuel avec une meilleure adaptation au visage
- Ajout de fonctionnalités d'animation pour visualiser les lunettes en mouvement

## Remerciements

Ce projet est basé sur le modèle [Hunyuan3D-2](https://github.com/Tencent/Hunyuan3D-2) développé par Tencent. Nous remercions les auteurs pour leur travail remarquable.
