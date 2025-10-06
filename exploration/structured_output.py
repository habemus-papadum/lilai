import marimo

__generated_with = "0.16.5"
app = marimo.App(width="full")


@app.cell
def _():
    return


@app.cell
def _():
    from openai import OpenAI

    client = OpenAI()
    return (client,)


@app.cell
def _():
    from pydantic import BaseModel
    return (BaseModel,)


@app.cell
def _(BaseModel, mo):
    class Item(BaseModel):
        name: str
        price: str

    class Menu(BaseModel):
        restaurant: str
        items: list[Item]
    mo.show_code()
    return (Menu,)


@app.cell
def _(Menu, client, mo):
    response = client.responses.parse(
        model="gpt-5-nano",
        input="What appetizers available at Kava on shawmut ave ?",
        tools=[{"type": "web_search"}],
        text_format=Menu,
    )
    mo.show_code()

    return (response,)


@app.cell
def _(response):
    res = response.output_parsed
    res
    return (res,)


@app.cell
def _(mo, res):
    mo.show_code(res.model_dump())
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    def menu(restaurant): 
        return """
        mezedakia
    small plates
    pikilia 8 each, 3 for 21
    cucumber yogurt dip, spicy feta spread, eggplant dip
    elies 9
    olives, olive oil, oregano
    feta psiti 14
    wrapped in phyllo, honey, sesame seeds
    bouyiourdi 15
    baked feta, cherry tomatoes, oregano, olive oil
    pita tis imeras 13
    pie of the day
    pantzaria 11
    baked beets, scordalia
    gigandes 11
    lima beans, tomato sauce, onion, carrots
    kolokithakia 19
    zucchini chips with tzatziki
    imam 15
    eggplant, tomato, onion, garlic, kasseri cheese
    gavros 10
    white anchovies, olive oil
    kalamarakia tia skaras 18 grilled calamari, capers, olives, mint, olive oil, lemon
    saganaki garides 17
    baked shrimp, peppers, onion, spicy tomatoes, feta
    oktapodi 19
    grilled octopus, olive oil, lemon, oregano
    lahano dolmades 16
    stuffed cabbage, rice, beef, herbs, olive oil, lemon
    hilopites 18
    lamb stew, onions, herbs, tomato broth
    stifado 18
    beef short rib, cipollini onion, demi glaze
    keftedes 16
    lamb meatballs
    souvlaki pork or chicken 13
    marinated grilled skewers
    loukaniko 15
    grilled greek sausage

    salata
    horiatiki 16
    tomato, cucumber, bell pepper, red onion,
    olives, capers, feta cheese, oregano, olive oil
    roka 14
    arugula, pear, walnuts, grilled halloumi,
    lemon dressing
    maroulosalata 15
    romaine, herbs, olives, feta cheese, yogurt vinaigrette
    sinodeftika
    sides 10
    greek fries
    feta, oregano
    bamies
    okra, fresh tomato, herbs, garlic
    spanakorizo
    rice, spinach, herbs, lemon, olive oil
    kounoupidi
    oven roasted cauliflower, spicy tomato
    patates lemonates
    potatoes, lemon, oregano, olive oil

    *Denotes raw or undercooked product. Massachusetts Law requires us to inform you that consuming raw or undercooked meat, poultry, seafood, shellfish, or eggs may increase your risk of food
    
        """

    mo.show_code()
    return (menu,)


@app.cell
def _(mo):
    tools = [
        {
            "type": "function",
            "name": "get_menu",
            "description": "Get the menu for a restaurant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "restaurant": {
                        "type": "string",
                        "description": "Name of restaurant",
                    },
                },
                "required": ["restaurant"],
            },
        },
    ]
    mo.show_code()
    return (tools,)


@app.cell
def _():
    return


@app.cell
def _():
    import json
    return (json,)


@app.cell
def _(Menu, client, json, menu, mo, tools):
    def function_call():
        input_list = [
            {"role": "user", "content": "What appetizers available at Kava?"}
        ]
        response = client.responses.parse(
            model="gpt-5-nano",
            input=input_list,
            tools=tools,
            text_format=Menu,
        )
        input_list += response.output

        for item in response.output:
            if item.type == "function_call":
                if item.name == "get_menu":
                
                    menu_text = menu(json.loads(item.arguments))
                
                    # 4. Provide function call results to the model
                    input_list.append({
                        "type": "function_call_output",
                        "call_id": item.call_id,
                        "output": json.dumps({
                          "menu": menu_text
                        })
                    })


        response = client.responses.parse(
            model="gpt-5-nano",
            input=input_list,
            tools=tools,
            text_format=Menu,
        )
    
        # 5. The model should be able to give a response!
        return response
    mo.show_code()
    return (function_call,)


@app.cell
def _(function_call, mo):
    r = function_call()
    mo.show_code()
    return (r,)


@app.cell
def _(r):
    r.output_parsed.model_dump()

    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
