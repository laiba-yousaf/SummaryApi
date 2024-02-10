from flask import Flask, request, jsonify
from flask_cors import CORS
from summary_generate_code import summarization_model, split_text, generate_predictions

app = Flask(__name__)
CORS(app)

# Initialize tokenizer and model
tokenizer, model = summarization_model()

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.json
        segments_list = data.get('segments', [])

        # Process each segment and create a list of summaries
        summaries_list = []
        for segment in segments_list:
            parts = split_text(segment)
            part_summaries = []
            for part in parts:
                summaries = generate_predictions(part, tokenizer, model)
                part_summaries.extend(summaries)
            summaries_list.append(part_summaries)

        return jsonify({'summaries': summaries_list})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=5001)
