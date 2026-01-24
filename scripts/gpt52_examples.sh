#!/bin/bash
#
# GPT-5.2 ZEH Failure Examples
# =============================
#
# This script demonstrates that GPT-5.2 makes errors on surprisingly small
# problem instances. Each example shows the Zero-Error Horizon (ZEH) where
# the model first fails.
#
# Requirements:
#   - OPENAI_API_KEY environment variable
#   - jq for JSON parsing
#

if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    exit 1
fi

echo "GPT-5.2 ZEH Failure Examples"
echo "============================="
echo ""
echo "MULTIPLICATION"
echo "  ZEH=126: GPT-5.2 fails at 127*82 (should be 10414)"
echo "  Input: 127*82="
echo "  Expected: 10414"
echo -n "  Got: "
curl -s https://api.openai.com/v1/responses -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" -d '{"model": "gpt-5.2-2025-12-11", "instructions": "Answer with only the integer.", "input": "127*82=", "max_output_tokens": 32, "temperature": 0}' | jq -r '.output[0].content[0].text'
echo ""

echo "PARITY"
echo "  ZEH=4: GPT-5.2 fails at 5-bit string '11000' (parity should be 0)"
echo "  Input: 11000"
echo "  Expected: 0"
echo -n "  Got: "
curl -s https://api.openai.com/v1/responses -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" -d '{"model": "gpt-5.2-2025-12-11", "instructions": "Compute the parity (XOR) of the binary string. Answer with only 0 or 1.", "input": "11000", "max_output_tokens": 32, "temperature": 0}' | jq -r '.output[0].content[0].text'
echo ""

echo "PARENTHESES"
echo "  ZEH=10: GPT-5.2 fails at 11-char string '((((())))))'  (unbalanced)"
echo "  Input: ((((())))))"
echo "  Expected: No"
echo -n "  Got: "
curl -s https://api.openai.com/v1/responses -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" -d '{"model": "gpt-5.2-2025-12-11", "instructions": "Is the parentheses string balanced? Answer with only Yes or No.", "input": "((((())))))", "max_output_tokens": 32, "temperature": 0}' | jq -r '.output[0].content[0].text'
echo ""

echo "GRAPH_COLORING"
echo "  ZEH=4: GPT-5.2 fails at 5-vertex graph (chromatic number should be 2)"
echo "  Input: Graph with 5 vertices and edges (1,2), (1,4), (1,5), (2,3)."
echo "  Expected: 2"
echo -n "  Got: "
curl -s https://api.openai.com/v1/responses -H "Authorization: Bearer $OPENAI_API_KEY" -H "Content-Type: application/json" -d '{"model": "gpt-5.2-2025-12-11", "instructions": "What is the chromatic number of this graph? Answer with only the integer.", "input": "Graph with 5 vertices and edges (1,2), (1,4), (1,5), (2,3).", "max_output_tokens": 32, "temperature": 0}' | jq -r '.output[0].content[0].text'
echo ""

echo "============================="
echo "These examples show that even state-of-the-art models"
echo "make errors on simple tasks with small inputs."