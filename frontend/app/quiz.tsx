import { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  ActivityIndicator,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import { useRouter } from 'expo-router';
import axios from 'axios';

const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'https://nutrition-launch.preview.emergentagent.com';

export default function Quiz() {
  const [question, setQuestion] = useState<any>(null);
  const [selectedAnswer, setSelectedAnswer] = useState<number | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const { token } = useAuth();
  const router = useRouter();

  useEffect(() => {
    loadDailyQuiz();
  }, []);

  const loadDailyQuiz = async () => {
    try {
      const response = await axios.get(
        `${BACKEND_URL}/api/quiz/daily`,
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setQuestion(response.data);
    } catch (error) {
      console.error('Error loading quiz:', error);
    } finally {
      setLoading(false);
    }
  };

  const submitAnswer = async (answerIndex: number) => {
    setSelectedAnswer(answerIndex);
    setLoading(true);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/api/quiz/answer`,
        {
          question_id: question.id,
          answer: answerIndex.toString(),
        },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      setResult(response.data);
    } catch (error) {
      console.error('Error submitting answer:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading && !question) {
    return (
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color="#4CAF50" />
          <Text style={styles.loadingText}>Loading quiz...</Text>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <ScrollView contentContainerStyle={styles.content}>
        <View style={styles.header}>
          <Ionicons name="school" size={48} color="#FFD700" />
          <Text style={styles.title}>Daily Food Quiz</Text>
          <Text style={styles.subtitle}>Test your nutrition knowledge!</Text>
        </View>

        {!result ? (
          <View style={styles.questionContainer}>
            <Text style={styles.question}>{question?.question}</Text>

            <View style={styles.optionsContainer}>
              {question?.options.map((option: string, index: number) => (
                <TouchableOpacity
                  key={index}
                  style={[
                    styles.optionButton,
                    selectedAnswer === index && styles.optionButtonSelected,
                  ]}
                  onPress={() => submitAnswer(index)}
                  disabled={loading}
                >
                  <View style={styles.optionNumber}>
                    <Text style={styles.optionNumberText}>
                      {String.fromCharCode(65 + index)}
                    </Text>
                  </View>
                  <Text style={styles.optionText}>{option}</Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>
        ) : (
          <View style={styles.resultContainer}>
            <View
              style={[
                styles.resultBadge,
                { backgroundColor: result.correct ? '#4CAF50' : '#FF5252' },
              ]}
            >
              <Ionicons
                name={result.correct ? 'checkmark-circle' : 'close-circle'}
                size={64}
                color="#fff"
              />
              <Text style={styles.resultTitle}>
                {result.correct ? 'Correct!' : 'Not Quite!'}
              </Text>
            </View>

            <View style={styles.statsContainer}>
              <View style={styles.statBox}>
                <Ionicons name="flame" size={24} color="#FFD700" />
                <Text style={styles.statValue}>{result.quiz_streak}</Text>
                <Text style={styles.statLabel}>Streak</Text>
              </View>
              <View style={styles.statBox}>
                <Ionicons name="star" size={24} color="#4CAF50" />
                <Text style={styles.statValue}>+{result.xp_earned}</Text>
                <Text style={styles.statLabel}>XP Earned</Text>
              </View>
              <View style={styles.statBox}>
                <Ionicons name="analytics" size={24} color="#2196F3" />
                <Text style={styles.statValue}>{result.accuracy}%</Text>
                <Text style={styles.statLabel}>Accuracy</Text>
              </View>
            </View>

            <View style={styles.explanationContainer}>
              <Text style={styles.explanationTitle}>💡 Learn More:</Text>
              <Text style={styles.explanationText}>{result.explanation}</Text>
            </View>

            <TouchableOpacity
              style={styles.doneButton}
              onPress={() => router.back()}
            >
              <Text style={styles.doneButtonText}>Done</Text>
            </TouchableOpacity>

            <Text style={styles.comeTomorrow}>Come back tomorrow for a new question!</Text>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0c0c0c',
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#888',
    marginTop: 16,
    fontSize: 16,
  },
  content: {
    padding: 24,
  },
  header: {
    alignItems: 'center',
    marginBottom: 40,
  },
  title: {
    fontSize: 28,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 16,
  },
  subtitle: {
    fontSize: 16,
    color: '#888',
    marginTop: 8,
  },
  questionContainer: {
    marginBottom: 32,
  },
  question: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#fff',
    lineHeight: 32,
    marginBottom: 32,
    textAlign: 'center',
  },
  optionsContainer: {
    gap: 16,
  },
  optionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    borderWidth: 2,
    borderColor: '#333',
  },
  optionButtonSelected: {
    borderColor: '#4CAF50',
    backgroundColor: '#1a3a1a',
  },
  optionNumber: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#4CAF50',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 16,
  },
  optionNumberText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  optionText: {
    flex: 1,
    fontSize: 16,
    color: '#fff',
  },
  resultContainer: {
    alignItems: 'center',
  },
  resultBadge: {
    width: '100%',
    padding: 32,
    borderRadius: 16,
    alignItems: 'center',
    marginBottom: 32,
  },
  resultTitle: {
    fontSize: 32,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 16,
  },
  statsContainer: {
    flexDirection: 'row',
    width: '100%',
    marginBottom: 32,
    gap: 12,
  },
  statBox: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
  },
  statValue: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 12,
    color: '#888',
    marginTop: 4,
  },
  explanationContainer: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 20,
    width: '100%',
    marginBottom: 32,
    borderLeftWidth: 4,
    borderLeftColor: '#4CAF50',
  },
  explanationTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 12,
  },
  explanationText: {
    fontSize: 15,
    color: '#ccc',
    lineHeight: 22,
  },
  doneButton: {
    backgroundColor: '#4CAF50',
    borderRadius: 12,
    padding: 16,
    width: '100%',
    alignItems: 'center',
    marginBottom: 16,
  },
  doneButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  comeTomorrow: {
    fontSize: 14,
    color: '#888',
    textAlign: 'center',
  },
});
