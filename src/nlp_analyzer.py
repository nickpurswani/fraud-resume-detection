"""
NLP Analyzer for Fraudulent Candidate Detection Tool

This module provides comprehensive NLP analysis capabilities including:
- Text preprocessing and normalization
- Named Entity Recognition (NER)
- Text similarity analysis
- Plagiarism detection
- Writing style analysis
- Sentiment analysis
- Keyword extraction
"""

import re
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModel
import textstat
from textblob import TextBlob
from fuzzywuzzy import fuzz
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher
import dateparser
from datetime import datetime, timedelta
from google import genai

# Configure logging
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('maxent_ne_chunker', quiet=True)
    nltk.download('words', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
except Exception as e:
    logger.warning(f"Error downloading NLTK data: {e}")

class NLPAnalyzer:
    """Comprehensive NLP analyzer for resume fraud detection"""

    def __init__(self, spacy_model: str = "en_core_web_sm",
                 sentence_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize NLP analyzer with specified models

        Args:
            spacy_model: spaCy model name
            sentence_model: Sentence transformer model name
        """
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(f"spaCy model '{spacy_model}' not found. Installing...")
            spacy.cli.download(spacy_model)
            self.nlp = spacy.load(spacy_model)

        self.sentence_model = SentenceTransformer(sentence_model)
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english')) | STOP_WORDS

        # Initialize Gemini client
        try:
            from config import config
            gemini_config = config.get_gemini_config()
            if gemini_config['api_key']:
                self.gemini_client = genai.Client(api_key=gemini_config['api_key'])
                self.gemini_model = gemini_config['model']
                self.gemini_config = gemini_config
            else:
                self.gemini_client = None
                logger.warning("Gemini API key not provided - advanced AI features will be limited")
        except Exception as e:
            logger.warning(f"Could not initialize Gemini client: {e}")
            self.gemini_client = None

        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline("sentiment-analysis")
        except Exception as e:
            logger.warning(f"Could not load sentiment analyzer: {e}")
            self.sentiment_analyzer = None

        # Initialize keyword extractor
        self.keyword_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedupLim=0.7,
            top=20
        )

    def preprocess_text(self, text: str, remove_stopwords: bool = True,
                       lemmatize: bool = True, lowercase: bool = True) -> str:
        """
        Preprocess text with various cleaning options

        Args:
            text: Input text
            remove_stopwords: Whether to remove stop words
            lemmatize: Whether to lemmatize words
            lowercase: Whether to convert to lowercase

        Returns:
            Preprocessed text
        """
        if not text or not isinstance(text, str):
            return ""

        # Basic cleaning
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-]', '', text)

        if lowercase:
            text = text.lower()

        # Tokenization and processing
        doc = self.nlp(text)
        tokens = []

        for token in doc:
            # Skip punctuation and spaces
            if token.is_punct or token.is_space:
                continue

            # Skip stop words if requested
            if remove_stopwords and token.lower_ in self.stop_words:
                continue

            # Lemmatize if requested
            if lemmatize:
                tokens.append(token.lemma_)
            else:
                tokens.append(token.text)

        return ' '.join(tokens)

    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Extract named entities from text

        Args:
            text: Input text

        Returns:
            Dictionary of entity types and their information
        """
        doc = self.nlp(text)
        entities = defaultdict(list)

        for ent in doc.ents:
            entities[ent.label_].append({
                'text': ent.text,
                'start': ent.start_char,
                'end': ent.end_char,
                'confidence': getattr(ent, 'confidence', 0.0)
            })

        return dict(entities)

    def extract_personal_info(self, text: str) -> Dict[str, Any]:
        """
        Extract personal information from resume text

        Args:
            text: Resume text

        Returns:
            Dictionary containing extracted personal information
        """
        info = {
            'names': [],
            'emails': [],
            'phones': [],
            'locations': [],
            'organizations': [],
            'dates': [],
            'urls': []
        }

        # Extract entities
        entities = self.extract_entities(text)

        # Process entities
        info['names'] = [e['text'] for e in entities.get('PERSON', [])]
        info['organizations'] = [e['text'] for e in entities.get('ORG', [])]
        info['locations'] = [e['text'] for e in entities.get('GPE', [])]

        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        info['emails'] = list(set(re.findall(email_pattern, text)))

        # Extract phone numbers
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            r'\(\d{3}\)\s*\d{3}[-.]?\d{4}',
            r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}'
        ]
        for pattern in phone_patterns:
            info['phones'].extend(re.findall(pattern, text))
        info['phones'] = list(set(info['phones']))

        # Extract URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+|www\.[^\s<>"{}|\\^`\[\]]+'
        info['urls'] = list(set(re.findall(url_pattern, text)))

        # Extract dates
        date_entities = entities.get('DATE', [])
        for date_entity in date_entities:
            try:
                parsed_date = dateparser.parse(date_entity['text'])
                if parsed_date:
                    info['dates'].append({
                        'text': date_entity['text'],
                        'parsed': parsed_date.isoformat(),
                        'year': parsed_date.year
                    })
            except:
                continue

        return info

    def extract_work_experience(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract work experience information

        Args:
            text: Resume text

        Returns:
            List of work experience entries
        """
        experiences = []

        # Common section headers for experience
        exp_patterns = [
            r'(?i)(professional\s+)?experience',
            r'(?i)work\s+history',
            r'(?i)employment\s+history',
            r'(?i)career\s+history'
        ]

        # Find experience section
        exp_section = ""
        for pattern in exp_patterns:
            match = re.search(rf'{pattern}.*?(?=\n[A-Z][A-Z\s]*:|\Z)', text, re.DOTALL)
            if match:
                exp_section = match.group(0)
                break

        if not exp_section:
            exp_section = text  # Use full text if no section found

        # Extract job entries (simplified approach)
        job_pattern = r'([A-Za-z\s&,]+)\s*[\|\-]\s*([A-Za-z\s&,\.]+)\s*[\|\-]\s*(\d{4}\s*[\-\–]\s*(?:\d{4}|present|current))'
        matches = re.findall(job_pattern, exp_section, re.IGNORECASE)

        for match in matches:
            title, company, duration = match
            experiences.append({
                'title': title.strip(),
                'company': company.strip(),
                'duration': duration.strip()
            })

        return experiences

    def extract_education(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract education information

        Args:
            text: Resume text

        Returns:
            List of education entries
        """
        education = []

        # Common section headers for education
        edu_patterns = [
            r'(?i)education',
            r'(?i)academic\s+background',
            r'(?i)qualifications'
        ]

        # Find education section
        edu_section = ""
        for pattern in edu_patterns:
            match = re.search(rf'{pattern}.*?(?=\n[A-Z][A-Z\s]*:|\Z)', text, re.DOTALL)
            if match:
                edu_section = match.group(0)
                break

        if not edu_section:
            edu_section = text  # Use full text if no section found

        # Extract degree information
        degree_pattern = r'(bachelor|master|phd|doctorate|associate|b\.?[as]\.?|m\.?[as]\.?|ph\.?d\.?)[^|]*\|?[^|]*(\d{4})'
        matches = re.findall(degree_pattern, edu_section, re.IGNORECASE)

        for match in matches:
            degree_type, year = match
            education.append({
                'degree': degree_type.strip(),
                'year': year.strip()
            })

        return education

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills from text

        Args:
            text: Resume text

        Returns:
            Dictionary categorizing different types of skills
        """
        skills = {
            'technical': [],
            'programming': [],
            'tools': [],
            'soft': [],
            'certifications': []
        }

        # Skill categories and keywords
        skill_categories = {
            'programming': [
                'python', 'java', 'javascript', 'c++', 'c#', 'php', 'ruby', 'go',
                'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'html', 'css'
            ],
            'technical': [
                'machine learning', 'ai', 'data science', 'deep learning',
                'neural networks', 'nlp', 'computer vision', 'blockchain',
                'cybersecurity', 'devops', 'microservices', 'api', 'rest'
            ],
            'tools': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
                'git', 'jira', 'confluence', 'slack', 'tableau', 'powerbi',
                'excel', 'photoshop', 'figma'
            ],
            'soft': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'analytical', 'creative', 'adaptable', 'organized'
            ]
        }

        text_lower = text.lower()

        # Extract skills based on categories
        for category, keywords in skill_categories.items():
            for keyword in keywords:
                if keyword in text_lower:
                    skills[category].append(keyword)

        # Extract skills from skills section
        skills_pattern = r'(?i)skills.*?(?=\n[A-Z][A-Z\s]*:|\Z)'
        skills_section_match = re.search(skills_pattern, text, re.DOTALL)

        if skills_section_match:
            skills_section = skills_section_match.group(0)
            # Additional skill extraction from dedicated section
            extracted_keywords = self.extract_keywords(skills_section)
            for keyword, score in extracted_keywords:
                if keyword.lower() not in [item.lower() for sublist in skills.values() for item in sublist]:
                    skills['technical'].append(keyword)

        return skills

    def calculate_text_similarity(self, text1: str, text2: str) -> Dict[str, float]:
        """
        Calculate various similarity metrics between two texts

        Args:
            text1: First text
            text2: Second text

        Returns:
            Dictionary of similarity scores
        """
        similarities = {}

        # Cosine similarity using sentence transformers
        try:
            embeddings = self.sentence_model.encode([text1, text2])
            similarities['semantic_similarity'] = float(
                cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            )
        except Exception as e:
            logger.warning(f"Error calculating semantic similarity: {e}")
            similarities['semantic_similarity'] = 0.0

        # Fuzzy string similarity
        similarities['fuzzy_ratio'] = fuzz.ratio(text1, text2) / 100.0
        similarities['fuzzy_partial'] = fuzz.partial_ratio(text1, text2) / 100.0
        similarities['fuzzy_token_sort'] = fuzz.token_sort_ratio(text1, text2) / 100.0
        similarities['fuzzy_token_set'] = fuzz.token_set_ratio(text1, text2) / 100.0

        # Sequence similarity
        similarities['sequence_similarity'] = SequenceMatcher(None, text1, text2).ratio()

        # TF-IDF similarity
        try:
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarities['tfidf_similarity'] = float(
                cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
            )
        except Exception as e:
            logger.warning(f"Error calculating TF-IDF similarity: {e}")
            similarities['tfidf_similarity'] = 0.0

        return similarities

    def detect_plagiarism(self, text: str, reference_texts: List[str],
                         threshold: float = 0.8) -> Dict[str, Any]:
        """
        Detect potential plagiarism in text

        Args:
            text: Text to check for plagiarism
            reference_texts: List of reference texts to compare against
            threshold: Similarity threshold for plagiarism detection

        Returns:
            Plagiarism detection results
        """
        results = {
            'is_plagiarized': False,
            'max_similarity': 0.0,
            'suspicious_matches': [],
            'similar_chunks': []
        }

        if not reference_texts:
            return results

        # Split text into chunks for detailed analysis
        sentences = sent_tokenize(text)

        for ref_text in reference_texts:
            ref_sentences = sent_tokenize(ref_text)

            # Calculate overall similarity
            similarity_scores = self.calculate_text_similarity(text, ref_text)
            max_sim = max(similarity_scores.values())

            if max_sim > results['max_similarity']:
                results['max_similarity'] = max_sim

            if max_sim >= threshold:
                results['is_plagiarized'] = True
                results['suspicious_matches'].append({
                    'similarity_score': max_sim,
                    'similarity_breakdown': similarity_scores,
                    'reference_text_preview': ref_text[:200] + '...'
                })

            # Check sentence-level similarity
            for i, sentence in enumerate(sentences):
                for j, ref_sentence in enumerate(ref_sentences):
                    sent_similarity = self.calculate_text_similarity(sentence, ref_sentence)
                    max_sent_sim = max(sent_similarity.values())

                    if max_sent_sim >= threshold:
                        results['similar_chunks'].append({
                            'original_sentence': sentence,
                            'reference_sentence': ref_sentence,
                            'similarity_score': max_sent_sim,
                            'sentence_index': i,
                            'reference_index': j
                        })

        return results

    def analyze_writing_style(self, text: str) -> Dict[str, Any]:
        """
        Analyze writing style characteristics

        Args:
            text: Text to analyze

        Returns:
            Writing style analysis results
        """
        analysis = {}

        # Basic text statistics
        analysis['word_count'] = len(word_tokenize(text))
        analysis['sentence_count'] = len(sent_tokenize(text))
        analysis['avg_words_per_sentence'] = (
            analysis['word_count'] / analysis['sentence_count']
            if analysis['sentence_count'] > 0 else 0
        )

        # Readability scores
        try:
            analysis['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
            analysis['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
            analysis['automated_readability_index'] = textstat.automated_readability_index(text)
            analysis['coleman_liau_index'] = textstat.coleman_liau_index(text)
            analysis['gunning_fog'] = textstat.gunning_fog(text)
        except Exception as e:
            logger.warning(f"Error calculating readability scores: {e}")
            analysis.update({
                'flesch_reading_ease': 0,
                'flesch_kincaid_grade': 0,
                'automated_readability_index': 0,
                'coleman_liau_index': 0,
                'gunning_fog': 0
            })

        # Linguistic features
        doc = self.nlp(text)

        # POS tag distribution
        pos_counts = Counter([token.pos_ for token in doc if not token.is_space])
        total_tokens = sum(pos_counts.values())
        analysis['pos_distribution'] = {
            pos: count / total_tokens for pos, count in pos_counts.items()
        }

        # Vocabulary richness (Type-Token Ratio)
        unique_words = set(token.lower_ for token in doc if token.is_alpha)
        analysis['type_token_ratio'] = len(unique_words) / analysis['word_count'] if analysis['word_count'] > 0 else 0

        # Average word length
        word_lengths = [len(token.text) for token in doc if token.is_alpha]
        analysis['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0

        return analysis

    def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of text

        Args:
            text: Text to analyze

        Returns:
            Sentiment analysis results
        """
        sentiment_results = {}

        # Using TextBlob
        try:
            blob = TextBlob(text)
            sentiment_results['textblob'] = {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except Exception as e:
            logger.warning(f"TextBlob sentiment analysis failed: {e}")
            sentiment_results['textblob'] = {'polarity': 0, 'subjectivity': 0}

        # Using Gemini AI (if available)
        if self.gemini_client:
            try:
                prompt = f"""
                Analyze the sentiment of the following resume text. Provide:
                1. Overall sentiment (POSITIVE, NEGATIVE, or NEUTRAL)
                2. Confidence score (0.0 to 1.0)
                3. Brief explanation

                Text: {text[:1000]}...

                Respond in JSON format: {{"overall_sentiment": "", "confidence": 0.0, "explanation": ""}}
                """

                response = self.gemini_client.models.generate_content(
                    model=self.gemini_model,
                    contents=prompt
                )

                # Parse JSON response
                import json
                try:
                    gemini_result = json.loads(response.text)
                    sentiment_results['gemini'] = {
                        'overall_sentiment': gemini_result.get('overall_sentiment', 'NEUTRAL'),
                        'confidence': float(gemini_result.get('confidence', 0.5)),
                        'explanation': gemini_result.get('explanation', ''),
                        'positive_ratio': 0.7 if gemini_result.get('overall_sentiment') == 'POSITIVE' else 0.3
                    }
                except json.JSONDecodeError:
                    # Fallback parsing if JSON fails
                    sentiment_results['gemini'] = {
                        'overall_sentiment': 'POSITIVE' if 'positive' in response.text.lower() else 'NEGATIVE' if 'negative' in response.text.lower() else 'NEUTRAL',
                        'confidence': 0.6,
                        'explanation': response.text[:200],
                        'positive_ratio': 0.5
                    }

            except Exception as e:
                logger.warning(f"Gemini sentiment analysis failed: {e}")
                sentiment_results['gemini'] = {
                    'overall_sentiment': 'NEUTRAL',
                    'confidence': 0,
                    'positive_ratio': 0.5
                }

        # Using Transformers (if available)
        if self.sentiment_analyzer:
            try:
                # Split long text into chunks
                chunks = [text[i:i+512] for i in range(0, len(text), 512)]
                sentiments = []

                for chunk in chunks:
                    result = self.sentiment_analyzer(chunk)
                    sentiments.append(result[0])

                # Aggregate results
                positive_scores = [s['score'] for s in sentiments if s['label'] == 'POSITIVE']
                negative_scores = [s['score'] for s in sentiments if s['label'] == 'NEGATIVE']

                sentiment_results['transformers'] = {
                    'overall_sentiment': 'POSITIVE' if len(positive_scores) > len(negative_scores) else 'NEGATIVE',
                    'confidence': np.mean([s['score'] for s in sentiments]),
                    'positive_ratio': len(positive_scores) / len(sentiments) if sentiments else 0
                }
            except Exception as e:
                logger.warning(f"Transformers sentiment analysis failed: {e}")
                sentiment_results['transformers'] = {
                    'overall_sentiment': 'NEUTRAL',
                    'confidence': 0,
                    'positive_ratio': 0.5
                }

        return sentiment_results

    def extract_keywords(self, text: str, max_keywords: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords from text

        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract

        Returns:
            List of (keyword, score) tuples
        """
        try:
            keywords = self.keyword_extractor.extract_keywords(text)
            return keywords[:max_keywords]
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []

    def detect_inconsistencies(self, text: str) -> Dict[str, Any]:
        """
        Detect various inconsistencies in the text

        Args:
            text: Text to analyze

        Returns:
            Dictionary of detected inconsistencies
        """
        inconsistencies = {
            'date_inconsistencies': [],
            'location_inconsistencies': [],
            'name_inconsistencies': [],
            'formatting_issues': [],
            'language_issues': []
        }

        # Extract personal info for consistency checking
        personal_info = self.extract_personal_info(text)

        # Check for multiple names
        if len(personal_info['names']) > 1:
            unique_names = set(personal_info['names'])
            if len(unique_names) > 1:
                inconsistencies['name_inconsistencies'].append({
                    'issue': 'Multiple different names found',
                    'names': list(unique_names)
                })

        # Check date consistency
        dates = personal_info['dates']
        if len(dates) > 1:
            years = [d['year'] for d in dates if 'year' in d]
            if years:
                year_diffs = [abs(years[i] - years[i-1]) for i in range(1, len(years))]
                if any(diff > 50 for diff in year_diffs):
                    inconsistencies['date_inconsistencies'].append({
                        'issue': 'Suspicious date gaps found',
                        'years': years
                    })

        # Check for formatting inconsistencies
        lines = text.split('\n')
        email_formats = set()
        for line in lines:
            if '@' in line:
                email_pattern = re.search(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', line)
                if email_pattern:
                    domain = email_pattern.group(0).split('@')[1]
                    email_formats.add(domain)

        if len(email_formats) > 2:
            inconsistencies['formatting_issues'].append({
                'issue': 'Multiple email domains found',
                'domains': list(email_formats)
            })

        return inconsistencies

    def generate_nlp_report(self, text: str, reference_texts: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive NLP analysis report

        Args:
            text: Text to analyze
            reference_texts: Optional reference texts for plagiarism detection

        Returns:
            Comprehensive NLP analysis report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'text_length': len(text),
            'analysis_results': {}
        }

        try:
            # Personal information extraction
            report['analysis_results']['personal_info'] = self.extract_personal_info(text)

            # Work experience
            report['analysis_results']['work_experience'] = self.extract_work_experience(text)

            # Education
            report['analysis_results']['education'] = self.extract_education(text)

            # Skills
            report['analysis_results']['skills'] = self.extract_skills(text)

            # Writing style analysis
            report['analysis_results']['writing_style'] = self.analyze_writing_style(text)

            # Sentiment analysis
            report['analysis_results']['sentiment'] = self.analyze_sentiment(text)

            # Keywords
            report['analysis_results']['keywords'] = self.extract_keywords(text)

            # Inconsistencies
            report['analysis_results']['inconsistencies'] = self.detect_inconsistencies(text)

            # Plagiarism detection (if reference texts provided)
            if reference_texts:
                report['analysis_results']['plagiarism'] = self.detect_plagiarism(text, reference_texts)

            report['status'] = 'success'

        except Exception as e:
            logger.error(f"Error generating NLP report: {e}")
            report['status'] = 'error'
            report['error_message'] = str(e)

        return report

if __name__ == "__main__":
    # Test the NLP analyzer
    analyzer = NLPAnalyzer()

    sample_text = """
    John Smith
    Senior Software Engineer
    Email: john.smith@email.com
    Phone: (555) 123-4567

    EXPERIENCE:
    Senior Software Engineer at Google (2020-Present)
    - Developed scalable web applications
    - Led a team of 10 engineers

    Software Engineer at Microsoft (2018-2020)
    - Built machine learning models
    - Improved system performance by 300%

    EDUCATION:
    Master of Science in Computer Science, Stanford University (2018)
    Bachelor of Science in Computer Science, UC Berkeley (2016)

    SKILLS:
    Python, Java, JavaScript, Machine Learning, AWS, Docker
    """

    report = analyzer.generate_nlp_report(sample_text)
    print("NLP Analysis Report:")
    for key, value in report['analysis_results'].items():
        print(f"{key}: {value}")
