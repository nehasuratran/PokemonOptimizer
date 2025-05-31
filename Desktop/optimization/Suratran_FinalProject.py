import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os
from statistics import mean
import random
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# Pokémon theme colors
POKEMON_COLORS = {
    'primary': '#FF0000',  # Pokémon Red
    'secondary': '#3B4CCA',  # Pokémon Blue
    'accent': '#FFDE00',  # Pikachu Yellow
    'background': '#FFFFFF',  # White
    'text': '#000000',  # Black
    'success': '#4CAF50',  # Green
    'warning': '#FFC107',  # Yellow
    'error': '#F44336',  # Red
    'type_colors': {
        'Normal': '#A8A878',
        'Fire': '#F08030',
        'Water': '#6890F0',
        'Electric': '#F8D030',
        'Grass': '#78C850',
        'Ice': '#98D8D8',
        'Fighting': '#C03028',
        'Poison': '#A040A0',
        'Ground': '#E0C068',
        'Flying': '#A890F0',
        'Psychic': '#F85888',
        'Bug': '#A8B820',
        'Rock': '#B8A038',
        'Ghost': '#705898',
        'Dragon': '#7038F8',
        'Dark': '#705848',
        'Steel': '#B8B8D0'
    }
}

def create_type_distribution_chart(deck):
    """Create a pie chart showing type distribution"""
    type_counts = {}
    for card in deck:
        if card["type"] not in type_counts:
            type_counts[card["type"]] = 0
        type_counts[card["type"]] += 1
    
    fig = go.Figure(data=[go.Pie(
        labels=list(type_counts.keys()),
        values=list(type_counts.values()),
        hole=.3,
        marker=dict(colors=[POKEMON_COLORS['type_colors'][t] for t in type_counts.keys()])
    )])
    
    fig.update_layout(
        title="Type Distribution",
        paper_bgcolor=POKEMON_COLORS['background'],
        plot_bgcolor=POKEMON_COLORS['background']
    )
    
    return fig

# Type effectiveness matrix
TYPE_EFFECTIVENESS = {
    'Normal': {'Rock': 0.5, 'Ghost': 0, 'Steel': 0.5},
    'Fire': {'Fire': 0.5, 'Water': 0.5, 'Grass': 2, 'Ice': 2, 'Bug': 2, 'Rock': 0.5, 'Dragon': 0.5, 'Steel': 2},
    'Water': {'Fire': 2, 'Water': 0.5, 'Grass': 0.5, 'Ground': 2, 'Rock': 2, 'Dragon': 0.5},
    'Electric': {'Water': 2, 'Electric': 0.5, 'Grass': 0.5, 'Ground': 0, 'Flying': 2, 'Dragon': 0.5},
    'Grass': {'Fire': 0.5, 'Water': 2, 'Grass': 0.5, 'Poison': 0.5, 'Ground': 2, 'Flying': 0.5, 'Bug': 0.5, 'Rock': 2, 'Dragon': 0.5, 'Steel': 0.5},
    'Ice': {'Fire': 0.5, 'Water': 0.5, 'Grass': 2, 'Ice': 0.5, 'Ground': 2, 'Flying': 2, 'Dragon': 2, 'Steel': 0.5},
    'Fighting': {'Normal': 2, 'Ice': 2, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 0.5, 'Bug': 0.5, 'Rock': 2, 'Ghost': 0, 'Dragon': 1, 'Dark': 2, 'Steel': 2},
    'Poison': {'Grass': 2, 'Poison': 0.5, 'Ground': 0.5, 'Rock': 0.5, 'Ghost': 0.5, 'Steel': 0},
    'Ground': {'Fire': 2, 'Electric': 2, 'Grass': 0.5, 'Poison': 2, 'Flying': 0, 'Bug': 0.5, 'Rock': 2, 'Steel': 2},
    'Flying': {'Electric': 0.5, 'Grass': 2, 'Fighting': 2, 'Bug': 2, 'Rock': 0.5, 'Steel': 0.5},
    'Psychic': {'Fighting': 2, 'Poison': 2, 'Psychic': 0.5, 'Dark': 0, 'Steel': 0.5},
    'Bug': {'Fire': 0.5, 'Grass': 2, 'Fighting': 0.5, 'Poison': 0.5, 'Flying': 0.5, 'Psychic': 2, 'Ghost': 0.5, 'Dark': 2, 'Steel': 0.5},
    'Rock': {'Fire': 2, 'Ice': 2, 'Fighting': 0.5, 'Ground': 0.5, 'Flying': 2, 'Bug': 2, 'Steel': 0.5},
    'Ghost': {'Normal': 0, 'Psychic': 2, 'Ghost': 2, 'Dark': 0.5},
    'Dragon': {'Dragon': 2, 'Steel': 0.5},
    'Dark': {'Fighting': 0.5, 'Psychic': 2, 'Ghost': 2, 'Dark': 0.5, 'Steel': 0.5},
    'Steel': {'Fire': 0.5, 'Water': 0.5, 'Electric': 0.5, 'Ice': 2, 'Rock': 2, 'Steel': 0.5}
}

def load_pokemon_cards():
    """Load Pokémon cards from CSV file"""
    # Read the local CSV file
    df = pd.read_csv("TOTALPOKEMON.csv")
    
    # Convert DataFrame to list of dictionaries
    cards = []
    for _, row in df.iterrows():
        card = {
            "name": row["name"],
            "type": row["type"],
            "hp": int(row["hp"]),
            "attack": int(row["attack"])
        }
        cards.append(card)
    
    return cards

def calculate_type_effectiveness(attacking_type, defending_type):
    """Calculate type effectiveness multiplier"""
    if attacking_type in TYPE_EFFECTIVENESS and defending_type in TYPE_EFFECTIVENESS[attacking_type]:
        return TYPE_EFFECTIVENESS[attacking_type][defending_type]
    return 1.0

def calculate_deck_rating(deck, card_pool):
    """Calculate a comprehensive deck rating based on multiple factors"""
    # Base stats
    total_hp = sum(card["hp"] for card in deck)
    total_attack = sum(card["attack"] for card in deck)
    
    # Calculate maximum possible values
    max_hp = max(card["hp"] for card in card_pool) * 7  # Maximum HP possible in a 7-card deck
    max_attack = max(card["attack"] for card in card_pool) * 7  # Maximum Attack possible in a 7-card deck
    
    # Type diversity score (0-1)
    type_diversity = len(set(card["type"] for card in deck)) / len(TYPE_EFFECTIVENESS)
    
    # Type coverage score
    type_coverage = 0
    for card in deck:
        for defending_type in TYPE_EFFECTIVENESS:
            if calculate_type_effectiveness(card["type"], defending_type) > 1:
                type_coverage += 1
    type_coverage = type_coverage / (len(TYPE_EFFECTIVENESS) * len(deck))
    
    # Balance score (how well distributed are the stats)
    balance_score = 1 - abs(total_hp - total_attack) / (total_hp + total_attack)
    
    # Final rating (weighted sum)
    rating = (
        0.3 * (total_hp / max_hp) +  # Normalized HP
        0.3 * (total_attack / max_attack) +  # Normalized Attack
        0.2 * type_diversity +  # Type diversity
        0.1 * type_coverage +  # Type coverage
        0.1 * balance_score  # Balance
    ) * 100  # Convert to percentage
    
    return {
        "rating": rating,
        "total_hp": total_hp,
        "total_attack": total_attack,
        "type_diversity": type_diversity,
        "type_coverage": type_coverage,
        "balance_score": balance_score,
        "max_possible_hp": max_hp,
        "max_possible_attack": max_attack
    }

def optimize_deck_static(card_pool, max_copies=2, min_types=4, randomize=True):
    """
    Optimize deck selection using static (lexicographic) optimization with randomization
    Returns the selected cards and their attributes
    """
    # Create model
    m = gp.Model("pokemon_deck_optimization_static")
    m.setParam('OutputFlag', 0)  # Suppress output
    
    # Get unique types
    types = list(set(card["type"] for card in card_pool))
    
    # Decision variables
    x = m.addVars(len(card_pool), vtype=GRB.BINARY, name="card_selected")
    y = m.addVars(len(types), vtype=GRB.BINARY, name="type_used")
    
    # Add randomization to objective coefficients if enabled
    if randomize:
        hp_noise = [random.uniform(0.95, 1.05) for _ in range(len(card_pool))]
        attack_noise = [random.uniform(0.95, 1.05) for _ in range(len(card_pool))]
    else:
        hp_noise = [1.0] * len(card_pool)
        attack_noise = [1.0] * len(card_pool)
    
    # Stage 1: Maximize total HP
    m.setObjective(gp.quicksum(card_pool[i]["hp"] * hp_noise[i] * x[i] for i in range(len(card_pool))), GRB.MAXIMIZE)
    
    # Add deck size constraint
    m.addConstr(gp.quicksum(x[i] for i in range(len(card_pool))) == 7, "deck_size")
    
    # Add card copy limit constraint
    for card_name in set(card["name"] for card in card_pool):
        card_indices = [i for i, card in enumerate(card_pool) if card["name"] == card_name]
        m.addConstr(gp.quicksum(x[i] for i in card_indices) <= max_copies, f"max_copies_{card_name}")
    
    # Add type tracking constraints
    for t, type_name in enumerate(types):
        m.addConstr(y[t] <= gp.quicksum(x[i] for i in range(len(card_pool)) 
                                       if card_pool[i]["type"] == type_name), f"type_used_{t}")
    
    # Add minimum type diversity as a soft constraint with penalty
    type_penalty = 1000  # Large penalty for not meeting minimum types
    slack = m.addVar(name="type_slack")
    m.addConstr(gp.quicksum(y[t] for t in range(len(types))) + slack >= min_types, "min_types")
    m.setObjective(m.getObjective() - type_penalty * slack)
    
    # Optimize for HP
    m.optimize()
    max_hp = m.objVal
    
    # Stage 2: Maximize attack given fixed HP
    m.setObjective(gp.quicksum(card_pool[i]["attack"] * attack_noise[i] * x[i] for i in range(len(card_pool))), GRB.MAXIMIZE)
    
    # Add HP constraint from previous stage with some flexibility
    hp_flexibility = 0.95  # Allow up to 5% reduction in HP
    m.addConstr(gp.quicksum(card_pool[i]["hp"] * x[i] for i in range(len(card_pool))) >= hp_flexibility * max_hp, "min_hp")
    
    # Optimize for attack
    m.optimize()
    
    # Get selected cards
    selected_cards = []
    for i in range(len(card_pool)):
        if x[i].X > 0.5:  # If card is selected
            selected_cards.append(card_pool[i])
    
    return selected_cards

def optimize_deck_adaptive(card_pool, weights=None, max_copies=2, min_types=4):
    """
    Optimize deck selection using adaptive (weighted) optimization
    Parameters:
    - card_pool: list of card dictionaries
    - weights: dictionary of weights for each objective (hp, attack, types)
    Returns the selected cards and their attributes
    """
    if weights is None:
        weights = {"hp": 0.4, "attack": 0.4, "types": 0.2}
    
    # Create model
    m = gp.Model("pokemon_deck_optimization_adaptive")
    m.setParam('OutputFlag', 0)  # Suppress output
    
    # Get unique types
    types = list(set(card["type"] for card in card_pool))
    
    # Decision variables
    x = m.addVars(len(card_pool), vtype=GRB.BINARY, name="card_selected")
    y = m.addVars(len(types), vtype=GRB.BINARY, name="type_used")
    
    # Add deck size constraint
    m.addConstr(gp.quicksum(x[i] for i in range(len(card_pool))) == 7, "deck_size")
    
    # Add card copy limit constraint
    for card_name in set(card["name"] for card in card_pool):
        card_indices = [i for i, card in enumerate(card_pool) if card["name"] == card_name]
        m.addConstr(gp.quicksum(x[i] for i in card_indices) <= max_copies, f"max_copies_{card_name}")
    
    # Add type tracking constraints
    for t, type_name in enumerate(types):
        m.addConstr(y[t] <= gp.quicksum(x[i] for i in range(len(card_pool)) 
                                       if card_pool[i]["type"] == type_name), f"type_used_{t}")
    
    # Add type diversity as a soft constraint with penalty
    type_penalty = weights["types"] * 2  # Scale penalty with type weight
    slack = m.addVar(name="type_slack")
    m.addConstr(gp.quicksum(y[t] for t in range(len(types))) + slack >= min_types, "min_types")
    
    # Calculate maximum possible values for normalization
    max_hp = max(card["hp"] for card in card_pool) * 7
    max_attack = max(card["attack"] for card in card_pool) * 7
    max_types = len(types)
    
    # Set weighted objective with type diversity penalty
    m.setObjective(
        weights["hp"] * (gp.quicksum(card_pool[i]["hp"] * x[i] for i in range(len(card_pool))) / max_hp) +
        weights["attack"] * (gp.quicksum(card_pool[i]["attack"] * x[i] for i in range(len(card_pool))) / max_attack) +
        weights["types"] * (gp.quicksum(y[t] for t in range(len(types))) / max_types) -
        type_penalty * slack,
        GRB.MAXIMIZE
    )
    
    # Optimize
    m.optimize()
    
    # Get selected cards
    selected_cards = []
    for i in range(len(card_pool)):
        if x[i].X > 0.5:  # If card is selected
            selected_cards.append(card_pool[i])
    
    return selected_cards

def create_deck_visualization(deck):
    """Create a visualization of the deck's stats and type distribution"""
    # Create stats radar chart with Pokémon colors
    stats = {
        "HP": sum(card["hp"] for card in deck),
        "Attack": sum(card["attack"] for card in deck),
        "Type Diversity": len(set(card["type"] for card in deck)),
        "Type Coverage": sum(1 for card in deck for t in TYPE_EFFECTIVENESS 
                           if calculate_type_effectiveness(card["type"], t) > 1) / len(TYPE_EFFECTIVENESS),
        "Balance": 1 - abs(sum(card["hp"] for card in deck) - sum(card["attack"] for card in deck)) / 
                  (sum(card["hp"] for card in deck) + sum(card["attack"] for card in deck))
    }
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=list(stats.values()),
        theta=list(stats.keys()),
        fill='toself',
        name='Deck Stats',
        fillcolor=POKEMON_COLORS['primary'],
        line=dict(color=POKEMON_COLORS['secondary'])
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, max(stats.values())],
                gridcolor=POKEMON_COLORS['accent'],
                linecolor=POKEMON_COLORS['secondary']
            ),
            bgcolor=POKEMON_COLORS['background']
        ),
        paper_bgcolor=POKEMON_COLORS['background'],
        plot_bgcolor=POKEMON_COLORS['background'],
        showlegend=False
    )
    
    return fig

def main():
    # Set page config with Pokémon theme
    st.set_page_config(
        page_title="Pokémon Deck Optimizer",
        page_icon="🎮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for Pokémon theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #FFFFFF;
        }
        .stButton>button {
            background-color: #FF0000;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 10px 20px;
            font-weight: bold;
        }
        .stButton>button:hover {
            background-color: #3B4CCA;
        }
        h1, h2, h3 {
            color: #FF0000;
        }
        .stSlider>div>div>div {
            background-color: #FFDE00;
        }
        .stDataFrame {
            background-color: #FFFFFF;
            border: 2px solid #3B4CCA;
            border-radius: 5px;
        }
        .stProgress>div>div>div {
            background-color: #FF0000;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Title with Pokémon styling
    st.markdown("<h1 style='text-align: center; color: #FF0000;'>Pokémon Deck Optimizer</h1>", unsafe_allow_html=True)
    
    # Load cards
    card_pool = load_pokemon_cards()
    
    # Sidebar controls with Pokémon styling
    with st.sidebar:
        st.markdown("<h2 style='color: #3B4CCA;'>Optimization Settings</h2>", unsafe_allow_html=True)
        method = st.selectbox("Optimization Method", ["Static", "Adaptive"])
        num_trials = st.slider("Number of Trials", 1, 20, 10)
        
        if method == "Adaptive":
            st.markdown("<h3 style='color: #3B4CCA;'>Weight Settings</h3>", unsafe_allow_html=True)
            hp_weight = st.slider("HP Weight", 0.1, 0.9, 0.4)
            attack_weight = st.slider("Attack Weight", 0.1, 0.9, 0.4)
            type_weight = st.slider("Type Weight", 0.1, 0.9, 0.2)
            weights = {"hp": hp_weight, "attack": attack_weight, "types": type_weight}
    
    # Run optimization
    if st.sidebar.button("Optimize Deck", key="optimize"):
        results = []
        
        with st.spinner("Optimizing your deck..."):
            for i in range(num_trials):
                if method == "Static":
                    deck = optimize_deck_static(card_pool, randomize=True)
                else:
                    deck = optimize_deck_adaptive(card_pool, weights)
                
                # Calculate deck rating
                rating = calculate_deck_rating(deck, card_pool)
                results.append({"deck": deck, "rating": rating})
        
        # Display results
        st.markdown("<h2 style='color: #FF0000;'>Optimization Results</h2>", unsafe_allow_html=True)
        
        # Show best deck
        best_deck = max(results, key=lambda x: x["rating"]["rating"])
        st.markdown(f"<h3 style='color: #3B4CCA;'>Best Deck (Rating: {best_deck['rating']['rating']:.1f}%)</h3>", 
                   unsafe_allow_html=True)
        
        # Create three columns for deck display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Display deck stats
            st.markdown("<h4 style='color: #3B4CCA;'>Deck Statistics</h4>", unsafe_allow_html=True)
            
            # Create a more detailed stats display
            stats = best_deck["rating"]
            st.markdown(f"""
                <div style='background-color: #FFFFFF; padding: 15px; border-radius: 10px; border: 2px solid #3B4CCA;'>
                    <h5 style='color: #FF0000;'>Overall Rating: {stats['rating']:.1f}%</h5>
                    <p><strong>HP:</strong> {stats['total_hp']} / {stats['max_possible_hp']} ({stats['total_hp']/stats['max_possible_hp']*100:.1f}%)</p>
                    <p><strong>Attack:</strong> {stats['total_attack']} / {stats['max_possible_attack']} ({stats['total_attack']/stats['max_possible_attack']*100:.1f}%)</p>
                    <p><strong>Type Diversity:</strong> {stats['type_diversity']*100:.1f}%</p>
                    <p><strong>Type Coverage:</strong> {stats['type_coverage']*100:.1f}%</p>
                    <p><strong>Balance Score:</strong> {stats['balance_score']*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Display deck visualization
            st.markdown("<h4 style='color: #3B4CCA;'>Deck Visualization</h4>", unsafe_allow_html=True)
            fig = create_deck_visualization(best_deck["deck"])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Display deck composition
            st.markdown("<h4 style='color: #3B4CCA;'>Deck Composition</h4>", unsafe_allow_html=True)
            deck_df = pd.DataFrame(best_deck["deck"])
            st.dataframe(deck_df, use_container_width=True)
            
            # Display type distribution
            st.markdown("<h4 style='color: #3B4CCA;'>Type Distribution</h4>", unsafe_allow_html=True)
            type_fig = create_type_distribution_chart(best_deck["deck"])
            st.plotly_chart(type_fig, use_container_width=True)
        
        with col3:
            # Display type effectiveness
            st.markdown("<h4 style='color: #3B4CCA;'>Type Effectiveness</h4>", unsafe_allow_html=True)
            type_effectiveness = {}
            for card in best_deck["deck"]:
                for defending_type in TYPE_EFFECTIVENESS:
                    if calculate_type_effectiveness(card["type"], defending_type) > 1:
                        if defending_type not in type_effectiveness:
                            type_effectiveness[defending_type] = []
                        type_effectiveness[defending_type].append(card["name"])
            
            for defending_type, attackers in type_effectiveness.items():
                st.markdown(
                    f"<div style='background-color: {POKEMON_COLORS['type_colors'][defending_type]}; "
                    f"padding: 10px; border-radius: 5px; margin: 5px 0;'>"
                    f"<strong>{defending_type}</strong> is weak against: {', '.join(attackers)}"
                    f"</div>",
                    unsafe_allow_html=True
                )

if __name__ == "__main__":
    main() 