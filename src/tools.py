import json

def get_market_data(ticker: str):
    """Retrieves simulated market data for a given ticker."""
    data = {
        "NVDA": 130.00,
        "AAPL": 190.00,
        "MSFT": 400.00
    }

    return json.dumps({
        "ticker": ticker,
        "price": data.get(ticker, "Unknown")
    })


def calculator(expression: str):
    """Evaluates a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {e}"