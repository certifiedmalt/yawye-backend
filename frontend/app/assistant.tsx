import { useState, useRef, useEffect } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  ScrollView,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Alert,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { useAuth } from '../context/AuthContext';
import axios from 'axios';

const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL;

interface Message {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

export default function Assistant() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [loading, setLoading] = useState(false);
  const [acceptedDisclaimer, setAcceptedDisclaimer] = useState(false);
  const { token } = useAuth();
  const scrollViewRef = useRef<ScrollView>(null);

  useEffect(() => {
    if (acceptedDisclaimer && messages.length === 0) {
      // Add welcome message
      setMessages([
        {
          id: '1',
          role: 'assistant',
          content: "Hi! I'm your health education assistant. I can help you understand:\n\n• Food ingredients and nutrition\n• Ultra-processed foods (UPFs)\n• General healthy eating tips\n• How to use this app\n\nRemember: I provide educational information only, not medical advice. Always consult healthcare professionals for personal health decisions.\n\nWhat would you like to know?",
          timestamp: new Date(),
        },
      ]);
    }
  }, [acceptedDisclaimer]);

  const sendMessage = async () => {
    if (!inputText.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: inputText.trim(),
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInputText('');
    setLoading(true);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/api/assistant/chat`,
        {
          message: userMessage.content,
          conversation_history: messages.map(m => ({
            role: m.role,
            content: m.content,
          })),
        },
        { headers: { Authorization: `Bearer ${token}` } }
      );

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.data.response,
        timestamp: new Date(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error: any) {
      Alert.alert('Error', error.response?.data?.detail || 'Failed to get response');
    } finally {
      setLoading(false);
    }
  };

  if (!acceptedDisclaimer) {
    return (
      <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
        <View style={styles.disclaimerContainer}>
          <Ionicons name="alert-circle" size={64} color="#FFA726" />
          <Text style={styles.disclaimerTitle}>Health Information Disclaimer</Text>
          
          <ScrollView style={styles.disclaimerScroll}>
            <Text style={styles.disclaimerText}>
              <Text style={styles.bold}>IMPORTANT: Please Read Carefully</Text>
              {'\n\n'}
              This AI assistant provides <Text style={styles.bold}>educational information only</Text>. It is NOT:
              {'\n\n'}
              ❌ Medical advice or diagnosis{'\n'}
              ❌ A substitute for healthcare professionals{'\n'}
              ❌ Personalized health recommendations{'\n'}
              ❌ Treatment guidance{'\n'}
              {'\n'}
              <Text style={styles.bold}>Always consult qualified healthcare providers</Text> for:
              {'\n\n'}
              ✅ Personal health questions{'\n'}
              ✅ Medical diagnoses{'\n'}
              ✅ Treatment decisions{'\n'}
              ✅ Dietary restrictions{'\n'}
              ✅ Medication interactions{'\n'}
              {'\n'}
              <Text style={styles.bold}>What This Assistant CAN Help With:</Text>
              {'\n\n'}
              • General nutrition education{'\n'}
              • Understanding food ingredients{'\n'}
              • Learning about UPFs{'\n'}
              • App usage questions{'\n'}
              {'\n'}
              <Text style={styles.bold}>Never ignore or delay professional medical advice</Text> based on information from this assistant.
            </Text>
          </ScrollView>

          <TouchableOpacity
            style={styles.acceptButton}
            onPress={() => setAcceptedDisclaimer(true)}
          >
            <Text style={styles.acceptButtonText}>I Understand - Continue</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top', 'bottom']}>
      <KeyboardAvoidingView
        style={styles.keyboardView}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        <View style={styles.header}>
          <Ionicons name="chatbubbles" size={24} color="#4CAF50" />
          <Text style={styles.headerTitle}>Health Assistant</Text>
          <TouchableOpacity onPress={() => {
            Alert.alert(
              'Disclaimer',
              'This assistant provides educational information only, not medical advice. Always consult healthcare professionals for personal health decisions.',
              [{ text: 'OK' }]
            );
          }}>
            <Ionicons name="information-circle-outline" size={24} color="#888" />
          </TouchableOpacity>
        </View>

        <ScrollView
          ref={scrollViewRef}
          style={styles.messagesContainer}
          contentContainerStyle={styles.messagesContent}
          onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}
        >
          {messages.map((message) => (
            <View
              key={message.id}
              style={[
                styles.messageBubble,
                message.role === 'user' ? styles.userMessage : styles.assistantMessage,
              ]}
            >
              {message.role === 'assistant' && (
                <View style={styles.assistantIcon}>
                  <Ionicons name="sparkles" size={16} color="#4CAF50" />
                </View>
              )}
              <Text
                style={[
                  styles.messageText,
                  message.role === 'user' ? styles.userMessageText : styles.assistantMessageText,
                ]}
              >
                {message.content}
              </Text>
            </View>
          ))}
          {loading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="small" color="#4CAF50" />
              <Text style={styles.loadingText}>Thinking...</Text>
            </View>
          )}
        </ScrollView>

        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            placeholder="Ask about nutrition, ingredients, or the app..."
            placeholderTextColor="#666"
            value={inputText}
            onChangeText={setInputText}
            multiline
            maxLength={500}
            editable={!loading}
          />
          <TouchableOpacity
            style={[styles.sendButton, (!inputText.trim() || loading) && styles.sendButtonDisabled]}
            onPress={sendMessage}
            disabled={!inputText.trim() || loading}
          >
            <Ionicons name="send" size={20} color={!inputText.trim() || loading ? '#666' : '#fff'} />
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0c0c0c',
  },
  keyboardView: {
    flex: 1,
  },
  disclaimerContainer: {
    flex: 1,
    padding: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  disclaimerTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#FFA726',
    marginTop: 16,
    marginBottom: 24,
    textAlign: 'center',
  },
  disclaimerScroll: {
    flex: 1,
    width: '100%',
    marginBottom: 24,
  },
  disclaimerText: {
    fontSize: 15,
    color: '#ccc',
    lineHeight: 24,
  },
  bold: {
    fontWeight: 'bold',
    color: '#fff',
  },
  acceptButton: {
    backgroundColor: '#4CAF50',
    borderRadius: 12,
    padding: 16,
    width: '100%',
    alignItems: 'center',
  },
  acceptButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
  },
  messagesContainer: {
    flex: 1,
  },
  messagesContent: {
    padding: 16,
  },
  messageBubble: {
    marginBottom: 16,
    borderRadius: 16,
    padding: 14,
    maxWidth: '85%',
  },
  userMessage: {
    alignSelf: 'flex-end',
    backgroundColor: '#4CAF50',
  },
  assistantMessage: {
    alignSelf: 'flex-start',
    backgroundColor: '#1a1a1a',
    flexDirection: 'row',
  },
  assistantIcon: {
    marginRight: 8,
    marginTop: 2,
  },
  messageText: {
    fontSize: 15,
    lineHeight: 22,
  },
  userMessageText: {
    color: '#fff',
  },
  assistantMessageText: {
    color: '#fff',
    flex: 1,
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    alignSelf: 'flex-start',
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 14,
    marginBottom: 16,
  },
  loadingText: {
    color: '#888',
    marginLeft: 8,
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 16,
    borderTopWidth: 1,
    borderTopColor: '#333',
    alignItems: 'flex-end',
  },
  input: {
    flex: 1,
    backgroundColor: '#1a1a1a',
    borderRadius: 20,
    paddingHorizontal: 16,
    paddingVertical: 12,
    color: '#fff',
    fontSize: 15,
    maxHeight: 100,
    marginRight: 12,
  },
  sendButton: {
    backgroundColor: '#4CAF50',
    width: 44,
    height: 44,
    borderRadius: 22,
    alignItems: 'center',
    justifyContent: 'center',
  },
  sendButtonDisabled: {
    backgroundColor: '#333',
  },
});
