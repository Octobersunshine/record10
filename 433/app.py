from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.exceptions import BadRequest
from sentiment_analyzer import (
    analyze_sentiment,
    analyze_sentiment_with_aspects,
    analyze_batch,
    get_available_domains,
)
from domain_lexicon import DOMAIN_NAMES
from trie_autocomplete import TrieAutocomplete

app = Flask(__name__)
CORS(app)

trie = TrieAutocomplete()


@app.errorhandler(BadRequest)
def handle_bad_request(e):
    if "Failed to decode" in str(e):
        return jsonify({
            "error": "Request body must be valid JSON"
        }), 400
    return jsonify({"error": str(e)}), 400


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "message": "Sentiment Analysis API is running"
    })


@app.route('/api/domains', methods=['GET'])
def list_domains():
    domains = get_available_domains()
    domain_list = []
    for d in domains:
        domain_list.append({
            "key": d,
            "name": DOMAIN_NAMES.get(d, d)
        })
    return jsonify({
        "count": len(domain_list),
        "domains": domain_list
    })


@app.route('/api/sentiment', methods=['POST'])
def sentiment():
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Request body must be JSON"
        }), 400

    text = data.get('text', '').strip()
    domain = data.get('domain', None)

    if not text:
        return jsonify({
            "error": "Field 'text' is required and cannot be empty"
        }), 400

    result = analyze_sentiment(text, domain=domain)
    if domain:
        result["domain"] = domain
        result["domain_name"] = DOMAIN_NAMES.get(domain, domain)
    return jsonify(result)


@app.route('/api/sentiment/aspects', methods=['POST'])
def sentiment_aspects():
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Request body must be JSON"
        }), 400

    text = data.get('text', '').strip()
    domain = data.get('domain', None)

    if not text:
        return jsonify({
            "error": "Field 'text' is required and cannot be empty"
        }), 400

    result = analyze_sentiment_with_aspects(text, domain=domain)
    if domain:
        result["domain"] = domain
        result["domain_name"] = DOMAIN_NAMES.get(domain, domain)
    return jsonify(result)


@app.route('/api/sentiment/batch', methods=['POST'])
def sentiment_batch():
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Request body must be JSON"
        }), 400

    texts = data.get('texts', [])
    domain = data.get('domain', None)

    if not isinstance(texts, list) or len(texts) == 0:
        return jsonify({
            "error": "Field 'texts' must be a non-empty list"
        }), 400

    if len(texts) > 100:
        return jsonify({
            "error": "Batch size cannot exceed 100 texts"
        }), 400

    results = analyze_batch(texts, domain=domain)
    return jsonify({
        "count": len(results),
        "domain": domain,
        "domain_name": DOMAIN_NAMES.get(domain, domain) if domain else None,
        "results": results
    })


@app.route('/api/autocomplete', methods=['GET'])
def autocomplete():
    prefix = request.args.get('prefix', '').strip()
    sort_by = request.args.get('sort_by', 'relevance')
    limit = request.args.get('limit', None)
    fuzzy = request.args.get('fuzzy', 'false').lower() == 'true'
    max_distance = request.args.get('max_distance', '1')
    segment = request.args.get('segment', 'false').lower() == 'true'
    include_score = request.args.get('include_score', 'true').lower() == 'true'

    if not prefix:
        return jsonify({
            "error": "Parameter 'prefix' is required and cannot be empty"
        }), 400

    if limit is not None:
        try:
            limit = int(limit)
        except ValueError:
            return jsonify({
                "error": "Parameter 'limit' must be a valid integer"
            }), 400
        if limit < 1:
            return jsonify({
                "error": "Parameter 'limit' must be a positive integer"
            }), 400
        if limit > trie.MAX_LIMIT:
            return jsonify({
                "error": f"Parameter 'limit' cannot exceed {trie.MAX_LIMIT}"
            }), 400

    try:
        max_distance = int(max_distance)
    except ValueError:
        return jsonify({
            "error": "Parameter 'max_distance' must be a valid integer"
        }), 400
    if max_distance < 0:
        return jsonify({
            "error": "Parameter 'max_distance' must be non-negative"
        }), 400
    if max_distance > trie.MAX_FUZZY_DISTANCE:
        return jsonify({
            "error": f"Parameter 'max_distance' cannot exceed {trie.MAX_FUZZY_DISTANCE}"
        }), 400

    if sort_by not in ['relevance', 'frequency', 'alphabetical']:
        return jsonify({
            "error": "Parameter 'sort_by' must be one of: 'relevance', 'frequency', 'alphabetical'"
        }), 400

    results = trie.autocomplete(
        prefix,
        sort_by=sort_by,
        limit=limit,
        fuzzy=fuzzy,
        max_distance=max_distance,
        segment=segment,
        include_score=include_score
    )

    response = {
        "prefix": prefix,
        "count": len(results),
        "sort_by": sort_by,
        "fuzzy": fuzzy,
        "max_distance": max_distance if fuzzy else 0,
        "segment": segment,
        "words": results
    }
    return jsonify(response)


@app.route('/api/words', methods=['POST'])
def add_word():
    data = request.get_json()

    if not data:
        return jsonify({
            "error": "Request body must be JSON"
        }), 400

    word = data.get('word', '').strip()
    frequency = data.get('frequency', 1)

    if not word:
        return jsonify({
            "error": "Field 'word' is required and cannot be empty"
        }), 400

    if not isinstance(frequency, int) or frequency < 1:
        return jsonify({
            "error": "Field 'frequency' must be a positive integer"
        }), 400

    success = trie.add_word(word, frequency)
    if success:
        return jsonify({
            "message": "Word added successfully",
            "word": word,
            "frequency": trie.get_frequency(word)
        })
    else:
        return jsonify({
            "error": "Failed to add word"
        }), 500


@app.route('/api/words', methods=['GET'])
def get_words():
    words = trie.get_all_words()
    return jsonify({
        "count": len(words),
        "words": words
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "available_endpoints": [
            "GET /health",
            "GET /api/domains",
            "POST /api/sentiment",
            "POST /api/sentiment/aspects",
            "POST /api/sentiment/batch",
            "GET /api/autocomplete?prefix=<prefix>&sort_by=<relevance|frequency|alphabetical>&limit=<number>&fuzzy=<true|false>&max_distance=<0-3>&segment=<true|false>",
            "GET /api/words",
            "POST /api/words"
        ]
    }), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({
        "error": f"Method {request.method} not allowed for this endpoint"
    }), 405


@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        "error": "Internal server error",
        "message": str(e)
    }), 500


if __name__ == '__main__':
    print("Starting Sentiment Analysis API server...")
    print("Health check: http://localhost:5000/health")
    print("List domains: GET http://localhost:5000/api/domains")
    print("API endpoint: POST http://localhost:5000/api/sentiment")
    print("Aspect endpoint: POST http://localhost:5000/api/sentiment/aspects")
    print("Batch endpoint: POST http://localhost:5000/api/sentiment/batch")
    print("Autocomplete endpoint: GET http://localhost:5000/api/autocomplete")
    print("Words endpoint: GET/POST http://localhost:5000/api/words")
    app.run(host='0.0.0.0', port=5000, debug=True)
