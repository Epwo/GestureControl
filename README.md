# Système de Contrôle par Gestes

Ce projet permet de contrôler votre ordinateur grâce à des gestes de la main détectés par une caméra, en utilisant la reconnaissance de formes avec MediaPipe.

## Installation

1. Clonez ce dépôt sur votre machine
2. Installez les dépendances requises :

```bash
pip install -r requirements.txt
```

## Configuration

1. Copiez le fichier `.env.sample` vers `.env`
2. Configurez la source vidéo dans le fichier `.env` :
   - `VIDEO_SOURCE = 0` pour la webcam intégrée
   - `VIDEO_SOURCE = 1` pour une webcam externe (ou autre index si plusieurs périphériques)

## Utilisation

Lancez l'application avec la commande :

```bash
python main.py
```

Une fenêtre s'ouvrira montrant le flux de la caméra avec le suivi de votre main.
Positionnez votre main droite devant la caméra pour commencer à utiliser les gestes.

## Gestes supportés

| Geste                | Description                                           | Action                        |
| -------------------- | ----------------------------------------------------- | ----------------------------- |
| 🤘 Signe des cornes  | Index et auriculaire relevés, autres doigts baissés | Lecture/Pause                 |
| ✋ Balayage vertical | Main ouverte avec mouvement vertical                  | Défilement haut/bas          |
| ✊ Main fermée      | Tous les doigts fermés avec le pouce caché          | Fermer l'application (Alt+F4) |
| ✌️ signe V      | Tous les doigts fermés sauf le majeur et l'index       | Appuyer sur OK/ENTREE |

## Comment ajouter de nouveaux gestes

Pour ajouter un nouveau geste :

1. Créez un nouveau fichier dans le dossier `src/gestures/`
2. Implémentez une fonction de détection semblable aux exemples existants
3. Importez et intégrez votre fonction dans `src/logic.py`

## Dépannage

- Si la caméra n'est pas détectée, vérifiez le paramètre `VIDEO_SOURCE` dans `.env`
- Pour quitter l'application, appuyez sur la touche 'q' lorsque la fenêtre de reconnaissance est active

## Remarques techniques

La détection est calibrée pour fonctionner avec la main droite uniquement. Le système utilise les points de repère de la main fournis par MediaPipe pour calculer les positions relatives des doigts et détecter les gestes spécifiques.

## Contribution

Ce projet a été effectué par le groupe 6 : composé de 
- POSTIC Ewann
- PIREAUD Nino
- MINVIELLE Julien
- BENARD Hugo
- DUCLOS Etienne
