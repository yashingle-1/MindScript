"""
MindScript - Sample Data Generator
==================================
Creates training dataset locally - no download needed!

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import os
import random

def create_training_data():
    """Create comprehensive training dataset for MindScript"""
    
    print("🧠 MindScript Data Generator")
    print("=" * 50)
    
    # Seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # Base personality profiles with sample texts
    base_profiles = [
        # ===== HIGH ANALYTICAL / LOW EMOTIONAL =====
        {
            "texts": [
                "I believe that every problem can be solved through logical analysis and systematic thinking. Breaking down complex issues into smaller components allows for methodical solutions. Data and evidence should always guide decisions rather than emotions or intuition.",
                "Mathematics has always fascinated me because of its precision and logic. There is beauty in finding elegant solutions to complex problems. I prefer structured approaches where outcomes can be predicted and measured accurately.",
                "When faced with a decision, I gather all available information and analyze it carefully. Emotional responses can cloud judgment, so I try to remain objective. Critical thinking is the most valuable skill anyone can develop.",
                "I enjoy debugging code because it requires systematic elimination of possibilities. Each error has a logical cause that can be identified through careful analysis. The satisfaction of finding the root cause is immensely rewarding.",
                "Scientific methodology provides the best framework for understanding the world. Hypotheses should be tested rigorously before drawing conclusions. I trust peer-reviewed research over anecdotal evidence.",
                "Spreadsheets and databases help me organize information efficiently. I track everything with metrics and measurements. Quantitative data reveals patterns that intuition might miss.",
                "Logic puzzles and brain teasers are my favorite pastimes. I enjoy the mental challenge of working through complex problems step by step. The process of reasoning is as satisfying as finding the solution.",
                "I approach relationships analytically too, trying to understand patterns and dynamics. Communication should be clear and unambiguous. Misunderstandings often arise from imprecise language.",
            ],
            "cOPN": 3.2, "cCON": 4.5, "cEXT": 2.5, "cAGR": 3.0, "cNEU": 2.0
        },
        
        # ===== HIGH CREATIVE / HIGH OPENNESS =====
        {
            "texts": [
                "The world is a canvas of infinite possibilities waiting to be explored. I love pushing boundaries and challenging conventional thinking. Innovation comes from questioning everything we take for granted.",
                "Art and music speak to my soul in ways words cannot express. I find inspiration everywhere - in nature, in conversations, in random moments of daily life. Creativity flows when I allow myself to be open to new experiences.",
                "I often daydream about solutions nobody has tried before. What if we approached this problem completely differently? The best ideas come from unexpected connections between unrelated concepts.",
                "Traditional rules sometimes need to be broken. I believe in experimentation and taking creative risks. Failure is just another step toward discovering something truly original.",
                "My imagination is my greatest asset. I can visualize possibilities that others might dismiss as impossible. Dreams and visions drive my creative process forward.",
                "I love brainstorming sessions where all ideas are welcome, no matter how wild. The craziest suggestions often lead to the most innovative solutions. Creativity thrives in an environment without judgment.",
                "Abstract concepts fascinate me more than concrete realities. I enjoy philosophical discussions about the nature of existence and consciousness. Exploring ideas is an adventure in itself.",
                "I express myself through multiple creative outlets - writing, painting, music, design. Each medium offers a different way to communicate emotions and ideas. Artistic expression is essential to my wellbeing.",
            ],
            "cOPN": 4.8, "cCON": 2.5, "cEXT": 3.8, "cAGR": 3.5, "cNEU": 3.0
        },
        
        # ===== HIGH SOCIAL / HIGH EXTRAVERSION =====
        {
            "texts": [
                "Nothing energizes me more than spending time with friends and meeting new people. Social gatherings are my favorite way to recharge. I thrive on human connection and meaningful conversations.",
                "I love being the center of attention at parties and events. Making people laugh brings me joy. Social situations where I can engage with many people simultaneously are exciting.",
                "Networking comes naturally to me because I genuinely enjoy learning about other people. Everyone has an interesting story to tell. Building relationships is one of life's greatest pleasures.",
                "Working alone for too long drains my energy. I need interaction with others to feel motivated and inspired. Collaborative environments bring out the best in me.",
                "I prefer talking through problems with others rather than thinking alone. Different perspectives help me see situations more clearly. Communication is the foundation of everything.",
                "My calendar is always full of social events and gatherings. I rarely turn down an invitation because every interaction is an opportunity for connection. Loneliness is my greatest fear.",
                "I express my thoughts openly and enjoy lively debates. Sharing ideas with others helps me refine my own thinking. Silence in social situations makes me uncomfortable.",
                "Team projects excite me because of the collaborative energy. Combining different strengths and perspectives leads to better outcomes. I love coordinating group activities and bringing people together.",
            ],
            "cOPN": 3.5, "cCON": 3.0, "cEXT": 4.9, "cAGR": 4.2, "cNEU": 2.5
        },
        
        # ===== HIGH STRUCTURED / HIGH CONSCIENTIOUSNESS =====
        {
            "texts": [
                "Organization is the foundation of success. I always plan my day in advance and follow a strict schedule. Discipline and consistency are more important than talent or inspiration.",
                "My workspace is always clean and organized. Everything has its place, and I become uncomfortable when things are messy or chaotic. Order brings clarity to my thinking.",
                "Deadlines are sacred to me, and I have never missed one. I often complete tasks well before they are due. Procrastination is a habit I simply cannot understand.",
                "I make detailed to-do lists and feel immense satisfaction checking items off. Goals should be specific, measurable, and time-bound. Vague aspirations rarely lead to concrete results.",
                "Reliability is my most valued trait. When I commit to something, people know they can count on me. My word is my bond, and I take promises seriously.",
                "I believe in working systematically rather than sporadically. Consistent effort over time produces better results than occasional bursts of activity. Patience and persistence pay off.",
                "Planning for the future gives me peace of mind. I have detailed five-year plans and regularly review my progress. Uncertainty is uncomfortable, so I prepare for contingencies.",
                "Quality matters more than speed. I prefer to do things correctly the first time rather than rush and make mistakes. Attention to detail distinguishes excellent work from mediocre work.",
            ],
            "cOPN": 2.8, "cCON": 4.9, "cEXT": 3.0, "cAGR": 3.2, "cNEU": 2.2
        },
        
        # ===== HIGH EMOTIONAL / HIGH NEUROTICISM =====
        {
            "texts": [
                "I feel everything so deeply and intensely. When someone I care about is hurting, I feel their pain as if it were my own. My emotions guide my decisions more than logic ever could.",
                "I often worry about the future and what might go wrong. Small problems can feel overwhelming when my anxiety takes over. Sleep sometimes escapes me when my mind is racing.",
                "Criticism affects me deeply, even when I know it's constructive. I replay difficult conversations in my head for days afterward. Rejection feels like a physical wound.",
                "My moods fluctuate throughout the day. I can feel on top of the world in the morning and completely defeated by evening. Emotional stability seems to come easily to others but not to me.",
                "I care so much about what others think of me. Social situations can be exhausting because I analyze every interaction afterward. Did I say the wrong thing? Did they like me?",
                "Stress manifests physically for me - headaches, tension, difficulty breathing. I feel emotions in my body before I can name them. Self-care is essential but often neglected.",
                "I am highly sensitive to the emotional atmosphere of any room. When tension exists between others, I absorb it even if I'm not involved. Conflict is extremely distressing.",
                "My empathy is both a gift and a burden. I understand others deeply but take on their emotional weight. Setting boundaries is something I constantly struggle with.",
            ],
            "cOPN": 3.2, "cCON": 2.8, "cEXT": 2.8, "cAGR": 4.0, "cNEU": 4.8
        },
        
        # ===== HIGH AGREEABLENESS / EMPATHETIC =====
        {
            "texts": [
                "Helping others brings me the greatest joy in life. I volunteer regularly and always offer support to those in need. Nothing is more fulfilling than making a positive difference.",
                "I avoid conflict whenever possible and strive to maintain harmony. Understanding different perspectives helps me find common ground. Compromise is not weakness but wisdom.",
                "Kindness is my guiding principle in all interactions. I believe that everyone is fighting battles we know nothing about. Compassion costs nothing but means everything.",
                "I trust people easily, perhaps too easily sometimes. I prefer to see the good in everyone rather than assume negative intentions. Cynicism seems like a sad way to live.",
                "Cooperation always produces better results than competition. I believe in lifting others up rather than stepping on them. Success is sweeter when shared with others.",
                "I struggle to say no when people ask for help. My needs often come last because I prioritize others. Self-sacrifice feels natural, though I know it's not always healthy.",
                "Forgiveness comes easily to me because holding grudges hurts the holder most. Everyone makes mistakes and deserves second chances. Peace of mind requires letting go.",
                "I genuinely care about others' wellbeing and check in regularly. Remembering birthdays and small details matters. Relationships require consistent nurturing and attention.",
            ],
            "cOPN": 3.5, "cCON": 3.5, "cEXT": 3.8, "cAGR": 4.8, "cNEU": 3.0
        },
        
        # ===== INTROVERTED / LOW EXTRAVERSION =====
        {
            "texts": [
                "I prefer spending time alone with my thoughts and creative projects. Solitude is restorative, not lonely. My inner world is rich and fulfilling enough.",
                "Large crowds and social events exhaust me quickly. I need time to recharge after social interactions. Small gatherings with close friends are preferable.",
                "I express myself better through writing than speaking. My thoughts become clearer when I have time to organize them. Speaking off the cuff is uncomfortable.",
                "Deep conversations with one person interest me more than surface-level chatter with many. Quality over quantity applies to relationships too. I have few but meaningful friendships.",
                "I observe more than I participate in social situations. Watching dynamics and listening carefully reveals more than constant talking. Silence is not awkward but comfortable.",
                "Working from home suits me perfectly. I am most productive when free from office interruptions. My own space allows for deep concentration.",
                "I often prefer the company of books, music, or nature over people. Animals are easier to be around than most humans. My perfect weekend involves solitude and reflection.",
                "Social media exhausts me with its constant noise. I prefer meaningful one-on-one communication. The pressure to be always available and responsive is draining.",
            ],
            "cOPN": 3.5, "cCON": 3.5, "cEXT": 1.8, "cAGR": 3.5, "cNEU": 3.2
        },
        
        # ===== ADVENTUROUS / HIGH OPENNESS + EXTRAVERSION =====
        {
            "texts": [
                "Travel and adventure are my greatest passions. I have visited over thirty countries and plan to see many more. Every new place teaches me something about myself.",
                "Routine bores me quickly. I need variety and novelty to stay engaged. The same schedule every day would drive me crazy within weeks.",
                "I embrace uncertainty rather than fear it. Not knowing what comes next is exciting. Life is too short for predictability and safety.",
                "Trying new foods, activities, and experiences brings me alive. I say yes to opportunities even when they scare me. Comfort zones are meant to be expanded.",
                "Spontaneous decisions often lead to the best memories. Some of my greatest adventures started with zero planning. I trust the journey to unfold naturally.",
                "Thrill-seeking activities like skydiving and bungee jumping appeal to me. The adrenaline rush is addictive. Fear is just excitement in disguise.",
                "I collect experiences rather than possessions. Material things mean little compared to memories and stories. My passport stamps are my treasures.",
                "Meeting people from different cultures enriches my understanding of humanity. Diversity makes the world beautiful and fascinating. Homogeneity is boring.",
            ],
            "cOPN": 4.9, "cCON": 2.2, "cEXT": 4.2, "cAGR": 3.5, "cNEU": 2.5
        },
        
        # ===== AMBITIOUS LEADER =====
        {
            "texts": [
                "I have always been driven to succeed and lead others. Taking charge comes naturally to me. I set ambitious goals and work tirelessly to achieve them.",
                "Competition motivates me to perform at my best. I want to be the best at everything I do. Second place is just the first loser.",
                "Leadership requires making tough decisions that others avoid. I am not afraid to take responsibility and face consequences. Accountability is essential.",
                "I believe in taking calculated risks for greater rewards. Playing it safe leads to mediocrity. Bold moves separate winners from followers.",
                "My career trajectory has been upward from the start. I sought promotions and opportunities aggressively. Ambition is not a dirty word.",
                "Time is my most valuable resource. I optimize every hour for maximum productivity. Wasting time on trivial matters is unacceptable.",
                "I surround myself with other ambitious people. Your network determines your net worth. Mediocrity is contagious, but so is excellence.",
                "Failure is just feedback that helps me improve. Setbacks motivate me to work harder. Resilience separates those who succeed from those who quit.",
            ],
            "cOPN": 3.8, "cCON": 4.5, "cEXT": 4.2, "cAGR": 2.5, "cNEU": 2.0
        },
        
        # ===== TECH ENTHUSIAST =====
        {
            "texts": [
                "Technology fascinates me endlessly. I spend hours learning about new frameworks and programming languages. The possibilities of artificial intelligence excite me tremendously.",
                "I love building things and solving technical challenges. Creating software that helps people is incredibly rewarding. Code is poetry in its own way.",
                "The future of technology is being written now, and I want to be part of it. Machine learning and AI are revolutionizing every industry. Staying current requires constant learning.",
                "I enjoy automating repetitive tasks because efficiency matters. Why do something manually when code can do it better? Laziness drives innovation.",
                "Open source communities inspire me with their collaborative spirit. Sharing knowledge freely benefits everyone. I contribute to projects whenever I can.",
                "Debugging is like solving a mystery. Each error has clues that lead to the solution. The satisfaction of fixing a stubborn bug is unmatched.",
                "I prefer communicating with machines sometimes. Computers are predictable and logical. Human interactions can be confusing and irrational.",
                "Technology should solve real problems, not create new ones. I focus on practical applications rather than theoretical concepts. Impact matters more than elegance.",
            ],
            "cOPN": 4.2, "cCON": 3.8, "cEXT": 2.5, "cAGR": 3.0, "cNEU": 2.5
        },
        
        # ===== NATURE LOVER =====
        {
            "texts": [
                "Nature is my sanctuary and source of peace. I find clarity in forests, mountains, and by the ocean. The natural world offers wisdom that modern life obscures.",
                "I believe humans have disconnected from the earth. Technology, while useful, creates distance from what truly matters. Simplicity and natural living appeal to me.",
                "Hiking and camping recharge me more than any vacation resort could. Sleeping under stars and waking with sunrise feels right. Outdoor adventures feed my soul.",
                "Environmental issues concern me deeply. Climate change threatens everything beautiful about our planet. Sustainable living is not optional but necessary.",
                "Animals are easier to be around than most people. Their authenticity and presence bring me comfort. I connect deeply with other species.",
                "Gardens and plants bring life to any space. Growing food connects me to the earth. Patience in nurturing plants teaches valuable lessons.",
                "Modern cities overwhelm me with noise and concrete. I dream of living somewhere rural and peaceful. Space and fresh air are essential needs.",
                "Natural beauty moves me emotionally. Sunsets, mountains, and oceans inspire awe and gratitude. We are small parts of something magnificent.",
            ],
            "cOPN": 4.0, "cCON": 3.2, "cEXT": 2.5, "cAGR": 4.2, "cNEU": 2.8
        },
        
        # ===== PHILOSOPHICAL THINKER =====
        {
            "texts": [
                "I often ponder the meaning of life and existence. Philosophy and deep conversations fascinate me endlessly. Why are we here? What is consciousness?",
                "These existential questions keep me awake at night. I seek wisdom and understanding above material success. The examined life is the only one worth living.",
                "I read extensively across philosophy, psychology, and spirituality. Different traditions offer unique perspectives on truth. Wisdom transcends cultural boundaries.",
                "Consciousness is the greatest mystery. How does subjective experience arise from physical matter? Understanding the mind would change everything.",
                "I question assumptions that others accept without thought. Conventional wisdom often turns out to be neither. Independent thinking requires courage.",
                "Death makes life meaningful. Awareness of mortality clarifies what truly matters. I try to live with this awareness rather than denying it.",
                "I enjoy mentoring others and sharing insights from my journey. Teaching deepens my own understanding. Knowledge grows when shared freely.",
                "Simple living allows more time for contemplation. Material excess distracts from what matters most. Clarity comes from removing rather than adding.",
            ],
            "cOPN": 4.8, "cCON": 2.8, "cEXT": 2.8, "cAGR": 3.5, "cNEU": 3.5
        },
    ]
    
    # Generate expanded dataset
    all_samples = []
    
    for profile in base_profiles:
        base_scores = {
            "cOPN": profile["cOPN"],
            "cCON": profile["cCON"],
            "cEXT": profile["cEXT"],
            "cAGR": profile["cAGR"],
            "cNEU": profile["cNEU"]
        }
        
        for text in profile["texts"]:
            # Add original
            sample = {"TEXT": text, **base_scores}
            all_samples.append(sample)
            
            # Add variations with noise
            for _ in range(8):
                noisy_scores = {
                    key: max(1.0, min(5.0, val + np.random.uniform(-0.4, 0.4)))
                    for key, val in base_scores.items()
                }
                all_samples.append({"TEXT": text, **noisy_scores})
    
    # Additional mixed personality texts
    mixed_samples = [
        {"TEXT": "I balance logic with intuition in my decisions. Sometimes analysis is needed, other times gut feelings guide me better. Both have their place.", "cOPN": 3.5, "cCON": 3.5, "cEXT": 3.5, "cAGR": 3.5, "cNEU": 3.0},
        {"TEXT": "Life has taught me to be adaptable. Some days require structure, others call for spontaneity. Flexibility is a strength.", "cOPN": 3.8, "cCON": 3.2, "cEXT": 3.5, "cAGR": 3.8, "cNEU": 3.2},
        {"TEXT": "I used to be more anxious but have learned coping strategies. Growth is possible at any age. Self-awareness is the first step to change.", "cOPN": 3.5, "cCON": 3.5, "cEXT": 3.0, "cAGR": 4.0, "cNEU": 3.5},
        {"TEXT": "My personality has different facets depending on context. At work I am focused and professional. With friends I am playful and relaxed.", "cOPN": 3.5, "cCON": 4.0, "cEXT": 3.8, "cAGR": 3.5, "cNEU": 2.8},
        {"TEXT": "I value both solitude and connection. Too much of either feels unbalanced. Knowing when to engage and when to withdraw is wisdom.", "cOPN": 3.8, "cCON": 3.5, "cEXT": 3.2, "cAGR": 3.8, "cNEU": 2.8},
    ]
    
    # Add mixed samples with variations
    for sample in mixed_samples:
        for _ in range(15):
            noisy = sample.copy()
            for key in ["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]:
                noisy[key] = max(1.0, min(5.0, sample[key] + np.random.uniform(-0.5, 0.5)))
            all_samples.append(noisy)
    
    # Create DataFrame
    df = pd.DataFrame(all_samples)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save to CSV
    output_path = "data/essays.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"\n✅ Dataset created successfully!")
    print(f"📁 Location: {output_path}")
    print(f"📊 Total samples: {len(df)}")
    print(f"\n📋 Columns: {list(df.columns)}")
    print(f"\n📈 Personality Score Statistics:")
    print(df[["cOPN", "cCON", "cEXT", "cAGR", "cNEU"]].describe().round(2))
    print(f"\n✨ Ready for training!")
    
    return df


if __name__ == "__main__":
    create_training_data()