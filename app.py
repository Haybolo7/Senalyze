from flask import Flask, render_template, request, jsonify
import nltk
from textstat import flesch_reading_ease

# Initialize Flask
app = Flask(__name__)

# Ensure NLTK data is present
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_simple_meaning(word):
    """A stable way to get meanings using NLTK's WordNet."""
    from nltk.corpus import wordnet
    syns = wordnet.synsets(word)
    if syns:
        return syns[0].definition()
    return "Definition not found"

def calculate_readability(text):
    if not text.strip(): return 1.0
    # Flesch Ease: 100 (Easy) to 0 (Hard)
    # We map it to your 1.0 - 5.0 scale
    score = flesch_reading_ease(text)
    scaled = 5.0 - (score / 25) 
    return round(max(1.0, min(5.0, scaled)), 1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        tokens = nltk.word_tokenize(text)
        tagged = nltk.pos_tag(tokens)
        
        analysis = []
        word_counts = {w.lower(): tokens.count(w) for w in tokens}

        for word, pos in tagged:
            if word.isalnum():
                meaning = get_simple_meaning(word)
            else:
                meaning = "Punctuation"

            analysis.append({
                "word": word,
                "pos": pos,
                "meaning": meaning,
                "count": len(word)
            })

        return jsonify({
            "analysis": analysis, 
            "readability": calculate_readability(text)
        })
    except Exception as e:
        print(f"Error: {e}") # This shows the error in VS Code terminal
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)