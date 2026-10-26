# ✦ Gemma 4 Studio — Interface Web d'IA Locale & Privée

**Gemma 4 Studio** est une interface web moderne, réactive et élégante pour faire tourner les modèles de la famille **Google Gemma 4** (Edge E2B, E4B, 12B, 26B MoE) directement sur votre machine en local, sans envoyer aucune donnée dans le cloud.

---

## ⚡ Deux Moteurs d'Exécution Locaux au Choix

| Moteur | Technologie | Cas d'usage recommandé | Prérequis |
| :--- | :--- | :--- | :--- |
| **⚡ WebGPU (In-Browser)** | `@mlc-ai/web-llm` | Modèles Edge ultra-légers (Gemma 4 E2B, Gemma 2 2B) s'exécutant **100% dans votre navigateur** | Navigateur avec WebGPU (Chrome, Edge, Firefox Nightly) |
| **🖥️ Serveur Local (Ollama)** | API REST / Streaming (`localhost:11434`) | Modèles plus volumineux (Gemma 4 E4B, 12B, 26B MoE, 31B) avec accélération CUDA/Metal/Vulkan | [Ollama](https://ollama.ai) ou [LM Studio](https://lmstudio.ai) |

---

## 🚀 Démarrage Rapide

### 1. Lancer l'interface web

```bash
# Dans le dossier webllm
npm install
npm run dev
```

L'interface sera accessible sur : **`http://localhost:5173`**

---

## 🛠️ Utilisation avec Ollama (Recommandé pour les grands modèles Gemma 4)

1. **Installer et lancer Ollama** :
   ```bash
   # Lancer le serveur avec autorisation CORS pour l'interface locale
   OLLAMA_ORIGINS="*" ollama serve
   ```

2. **Télécharger un modèle Gemma** :
   ```bash
   ollama run gemma4:latest
   # Ou variantes selon votre VRAM :
   # ollama run gemma4:e2b
   # ollama run gemma4:e4b
   # ollama run gemma4:12b
   # ollama run gemma2:2b
   ```

3. Dans l'interface web, sélectionnez l'onglet **🖥️ Ollama** :
   - Vos modèles installés sont **automatiquement détectés**.
   - Vous pouvez tester la connexion dans le menu **Paramètres (⚙️)**.

---

## 🧠 Fonctionnalités Clés de Gemma 4 Studio

- **Visualiseur du Thinking Mode (Raisonnement)** : Détection et rendu en direct des balises `<thought>` de Gemma 4 dans un accordéon repliable animé avec chrono et compteur de tokens de réflexion.
- **Support Multimodal (Vision)** : Glissez-déposez ou joignez une image pour analyse visuelle par les modèles multimodaux de Gemma 4.
- **Télémétrie en temps réel (HUD)** : Débit en tokens par seconde (tok/s), temps jusqu'au premier token (TTFT en ms), et nombre total de tokens.
- **Gestionnaire de conversations** : Sauvegarde automatique de l'historique dans `localStorage`, création de multiples sessions et suppression en 1 clic.
- **Export Markdown** : Exportez vos échanges et raisonnements complets au format `.md`.
- **Blocs de Code avec Coloration & Copie** : Mise en forme propre des réponses avec bouton de copie instantanée.
- **Ajustement des Hyperparamètres** : Température, Top-P, Tokens Max et présets d'instructions système (Codeur, Raisonneur, Synthétique, Polyvalent).
