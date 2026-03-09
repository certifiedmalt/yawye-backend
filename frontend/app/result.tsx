import { useEffect, useState, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
  Animated,
  Easing,
  Linking,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';
import ConfettiCannon from 'react-native-confetti-cannon';
import * as Haptics from 'expo-haptics';

interface HarmfulIngredient {
  name: string;
  health_impact: string;
  severity: string;
  processing_level?: string;
  research_summary: string;
  study_link?: string;
}

interface BeneficialIngredient {
  name: string;
  health_benefit: string;
  benefit_type?: string;
  key_nutrients?: string;
  processing_level?: string;
  research_summary: string;
  study_link?: string;
}

interface Analysis {
  harmful_ingredients: HarmfulIngredient[];
  beneficial_ingredients: BeneficialIngredient[];
  overall_score: number;
  upf_score?: string;
  processing_category?: string;
  recommendation: string;
}

interface ProductData {
  product_name: string;
  brands: string;
  ingredients_text: string;
  image_url: string;
  analysis: Analysis;
}

export default function Result() {
  const router = useRouter();
  const params = useLocalSearchParams();
  const [productData, setProductData] = useState<ProductData | null>(null);
  const [expandedResearch, setExpandedResearch] = useState<{ [key: string]: boolean }>({});
  const scoreAnim = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    if (productData) {
      // Animate the score ring whenever productData changes
      Animated.timing(scoreAnim, {
        toValue: productData.analysis.overall_score,
        duration: 800,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: false,
      }).start();
    }
  }, [productData]);

  const [showResearchModal, setShowResearchModal] = useState(false);
  const confettiRef = useRef<any>(null);

  useEffect(() => {
    if (params.productData) {
      try {
        const data = JSON.parse(params.productData as string);
        setProductData(data);
        
        // Celebrate healthy products!
        if (data.analysis.overall_score >= 8) {
          // Trigger confetti
          confettiRef.current?.start();
          // Haptic feedback
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
        } else if (data.analysis.overall_score <= 3) {
          // Warning haptic for unhealthy products
          Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
        }
      } catch (error) {
        console.error('Error parsing product data:', error);
      }
    }
  }, [params]);

  const toggleResearch = (ingredientName: string) => {
    setExpandedResearch(prev => ({
      ...prev,
      [ingredientName]: !prev[ingredientName]
    }));
  };

  if (!productData) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Loading...</Text>
      </View>
    );
  }

  const { product_name, brands, image_url, analysis } = productData;
  const scoreColor =
    analysis.overall_score >= 7
      ? '#00E676' // Bright vibrant green for healthy!
      : analysis.overall_score >= 4
      ? '#FFD54F' // Bright yellow for mediocre
      : '#FF5252'; // Red for unhealthy
  
  const scoreGradient = 
    analysis.overall_score >= 7
      ? ['#00E676', '#00C853'] // Green gradient for healthy
      : analysis.overall_score >= 4
      ? ['#FFD54F', '#FFA726'] // Yellow-orange gradient
      : ['#FF5252', '#D32F2F']; // Red gradient for unhealthy

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
      {/* Confetti for healthy products */}
      <ConfettiCannon
        ref={confettiRef}
        count={100}
        origin={{ x: -10, y: 0 }}
        autoStart={false}
        fadeOut
      />
      
      <ScrollView contentContainerStyle={styles.scrollContent}>
        {image_url && (
          <Image source={{ uri: image_url }} style={styles.productImage} />
        )}

        <View style={styles.header}>
          <Text style={styles.productName}>{product_name}</Text>
          <Text style={styles.brandName}>{brands}</Text>
        </View>

        <View style={[styles.scoreCard, { borderColor: scoreColor }]}>
          <Text style={styles.scoreLabel}>Health Score</Text>

          {/* Animated circular score ring */}
          <View style={styles.scoreRingContainer}>
            <View style={styles.scoreRingBackground} />
            <Animated.View
              style={[
                styles.scoreRingFill,
                {
                  borderColor: scoreColor,
                  transform: [
                    {
                      rotateZ: scoreAnim.interpolate({
                        inputRange: [0, 10],
                        outputRange: ['0deg', '360deg'],
                      }),
                    },
                  ],
                },
              ]}
            />
            <View style={styles.scoreRingInner}>
              <Text style={[styles.scoreValue, { color: scoreColor }]}>
                {analysis.overall_score}
              </Text>
              <Text style={styles.scoreOutOf}>/10</Text>
            </View>
          </View>
          {analysis.processing_category && (
            <View style={styles.processingBadge}>
              <Text style={styles.processingText}>{analysis.processing_category}</Text>
            </View>
          )}
          {analysis.upf_score && (
            <Text style={styles.upfScore}>UPF Content: {analysis.upf_score}</Text>
          )}
          <Text style={styles.recommendation}>{analysis.recommendation}</Text>
        </View>

        {analysis.harmful_ingredients && analysis.harmful_ingredients.length > 0 && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Ionicons name="warning" size={24} color="#FF5252" />
              <Text style={styles.sectionTitle}>Ingredients to Avoid</Text>
            </View>
            {analysis.harmful_ingredients.map((ingredient, index) => (
              <View key={index} style={styles.ingredientCard}>
                <View style={styles.ingredientHeader}>
                  <Text style={styles.ingredientName}>{ingredient.name}</Text>
                  <View
                    style={[
                      styles.severityBadge,
                      {
                        backgroundColor:
                          ingredient.severity === 'high'
                            ? '#FF5252'
                            : ingredient.severity === 'medium'
                            ? '#FFA726'
                            : '#FFEB3B',
                      },
                    ]}
                  >
                    <Text style={styles.severityText}>
                      {ingredient.severity.toUpperCase()}
                    </Text>
                  </View>
                </View>
                {ingredient.processing_level && (
                  <Text style={styles.processingLevel}>{ingredient.processing_level}</Text>
                )}
                <Text style={styles.healthImpact}>
                  {ingredient.health_impact}
                </Text>
                
                {/* Collapsible Research Section */}
                <TouchableOpacity 
                  style={styles.researchToggle}
                  onPress={() => toggleResearch(`harmful_${index}`)}
                >
                  <Ionicons name="information-circle-outline" size={18} color="#4CAF50" />
                  <Text style={styles.researchToggleText}>Research</Text>
                  <Ionicons 
                    name={expandedResearch[`harmful_${index}`] ? "chevron-up" : "chevron-down"} 
                    size={16} 
                    color="#4CAF50" 
                  />
                </TouchableOpacity>
                
                {expandedResearch[`harmful_${index}`] && (
                  <View style={styles.researchContent}>
                    <Text style={styles.researchText}>
                      {ingredient.research_summary || ingredient.concern || 'Research data not available for this ingredient. The health impact information above summarizes the key concerns.'}
                    </Text>
                    {ingredient.study_link && (
                      <TouchableOpacity 
                        onPress={() => Linking.openURL(ingredient.study_link!)}
                        style={styles.studyLink}
                      >
                        <Ionicons name="open-outline" size={14} color="#4CAF50" />
                        <Text style={styles.studyLinkText}>View Study</Text>
                      </TouchableOpacity>
                    )}
                  </View>
                )}
              </View>
            ))}
          </View>
        )}

        {analysis.beneficial_ingredients && analysis.beneficial_ingredients.length > 0 && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Ionicons name="checkmark-circle" size={24} color="#4CAF50" />
              <Text style={styles.sectionTitle}>Beneficial Ingredients</Text>
            </View>
            {analysis.beneficial_ingredients.map((ingredient, index) => (
              <View key={index} style={styles.ingredientCard}>
                <Text style={styles.ingredientName}>{ingredient.name}</Text>
                {ingredient.processing_level && (
                  <Text style={styles.processingLevel}>{ingredient.processing_level}</Text>
                )}
                <Text style={styles.healthImpact}>
                  {ingredient.health_benefit}
                </Text>
                
                {/* Collapsible Research Section */}
                <TouchableOpacity 
                  style={styles.researchToggle}
                  onPress={() => toggleResearch(`beneficial_${index}`)}
                >
                  <Ionicons name="information-circle-outline" size={18} color="#4CAF50" />
                  <Text style={styles.researchToggleText}>Research</Text>
                  <Ionicons 
                    name={expandedResearch[`beneficial_${index}`] ? "chevron-up" : "chevron-down"} 
                    size={16} 
                    color="#4CAF50" 
                  />
                </TouchableOpacity>
                
                {expandedResearch[`beneficial_${index}`] && (
                  <View style={styles.researchContent}>
                    {ingredient.key_nutrients && (
                      <Text style={styles.keyNutrients}>
                        Key nutrients: {ingredient.key_nutrients}
                      </Text>
                    )}
                    <Text style={styles.researchText}>
                      {ingredient.research_summary || ingredient.benefit || 'Research data not available for this ingredient. The health benefit information above summarizes the key advantages.'}
                    </Text>
                    {ingredient.study_link && (
                      <TouchableOpacity 
                        onPress={() => Linking.openURL(ingredient.study_link!)}
                        style={styles.studyLink}
                      >
                        <Ionicons name="open-outline" size={14} color="#4CAF50" />
                        <Text style={styles.studyLinkText}>View Study</Text>
                      </TouchableOpacity>
                    )}
                  </View>
                )}
              </View>
            ))}
          </View>
        )}

        {/* View All Research Button */}
        <TouchableOpacity
          style={styles.viewResearchButton}
          onPress={() => setShowResearchModal(true)}
        >
          <Ionicons name="book-outline" size={20} color="#4CAF50" />
          <Text style={styles.viewResearchText}>View All Research Studies</Text>
        </TouchableOpacity>

        <TouchableOpacity
          style={styles.scanAgainButton}
          onPress={() => router.replace('/main')}
        >
          <Ionicons name="home" size={24} color="#fff" />
          <Text style={styles.scanAgainText}>Return Home</Text>
        </TouchableOpacity>
      </ScrollView>

      {/* Research Modal */}
      {showResearchModal && (
        <View style={styles.modalOverlay}>
          <View style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>Research Studies</Text>
              <TouchableOpacity onPress={() => setShowResearchModal(false)}>
                <Ionicons name="close" size={28} color="#fff" />
              </TouchableOpacity>
            </View>
            
            <ScrollView style={styles.modalScroll}>
              {analysis.harmful_ingredients && analysis.harmful_ingredients.length > 0 && (
                <View style={styles.modalSection}>
                  <Text style={styles.modalSectionTitle}>⚠️ Harmful Ingredients Research</Text>
                  {analysis.harmful_ingredients.map((ingredient, index) => (
                    <View key={index} style={styles.modalItem}>
                      <Text style={styles.modalIngredientName}>{ingredient.name}</Text>
                      <Text style={styles.modalResearchText}>
                        {ingredient.research_summary || ingredient.concern || ingredient.health_impact || 'Research summary not available.'}
                      </Text>
                    </View>
                  ))}
                </View>
              )}
              
              {analysis.beneficial_ingredients && analysis.beneficial_ingredients.length > 0 && (
                <View style={styles.modalSection}>
                  <Text style={styles.modalSectionTitle}>✅ Beneficial Ingredients Research</Text>
                  {analysis.beneficial_ingredients.map((ingredient, index) => (
                    <View key={index} style={styles.modalItem}>
                      <Text style={styles.modalIngredientName}>{ingredient.name}</Text>
                      <Text style={styles.modalResearchText}>
                        {ingredient.research_summary || ingredient.health_benefit || ingredient.benefit || 'Research summary not available.'}
                      </Text>
                    </View>
                  ))}
                </View>
              )}
            </ScrollView>
            
            <TouchableOpacity 
              style={styles.modalCloseButton}
              onPress={() => setShowResearchModal(false)}
            >
              <Text style={styles.modalCloseButtonText}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0c0c0c',
  },
  scrollContent: {
    padding: 24,
  },
  productImage: {
    width: '100%',
    height: 200,
    borderRadius: 16,
    marginBottom: 24,
  },
  header: {
    marginBottom: 24,
  },
  productName: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  brandName: {
    fontSize: 16,
    color: '#888',
  },
  scoreCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 24,
    alignItems: 'center',
    marginBottom: 24,
    borderWidth: 2,
  },
  scoreLabel: {
    fontSize: 16,
    color: '#888',
    marginBottom: 8,
  },
  scoreValue: {
    fontSize: 48,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  recommendation: {
    fontSize: 16,
    color: '#fff',
    textAlign: 'center',
  },
  processingBadge: {
    backgroundColor: '#2196F3',
    paddingHorizontal: 16,
    paddingVertical: 6,
    borderRadius: 20,
    marginVertical: 8,
  },
  processingText: {
    color: '#fff',
    fontSize: 14,
    fontWeight: 'bold',
  },
  upfScore: {
    fontSize: 14,
    color: '#FFA726',
    marginTop: 8,
    fontWeight: '600',
  },
  processingLevel: {
    fontSize: 13,
    color: '#2196F3',
    marginBottom: 8,
    fontStyle: 'italic',
  },
  section: {
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#fff',
    marginLeft: 12,
  },
  ingredientCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  ingredientHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  ingredientName: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  severityBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 12,
  },
  severityText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#000',
  },
  ingredientDescription: {
    fontSize: 14,
    color: '#ccc',
    lineHeight: 20,
    marginBottom: 12,
  },
  healthImpact: {
    fontSize: 15,
    color: '#fff',
    lineHeight: 22,
    marginBottom: 12,
  },
  researchToggle: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    backgroundColor: '#0a0a0a',
    borderRadius: 8,
    marginTop: 4,
  },
  researchToggleText: {
    fontSize: 14,
    color: '#4CAF50',
    marginLeft: 6,
    marginRight: 6,
    fontWeight: '600',
  },
  researchContent: {
    backgroundColor: '#111',
    padding: 14,
    borderRadius: 8,
    marginTop: 10,
    borderLeftWidth: 3,
    borderLeftColor: '#4CAF50',
  },
  researchText: {
    fontSize: 13,
    color: '#aaa',
    lineHeight: 20,
  },
  viewResearchButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginTop: 16,
    marginBottom: 16,
    borderWidth: 1,
    borderColor: '#4CAF50',
  },
  viewResearchText: {
    color: '#4CAF50',
    fontSize: 16,
    fontWeight: '600',
    marginLeft: 8,
  },
  modalOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.9)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  modalContent: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    width: '100%',
    maxHeight: '90%',
    padding: 20,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 20,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#333',
  },
  modalTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    color: '#fff',
  },
  modalScroll: {
    maxHeight: '70%',
  },
  modalSection: {
    marginBottom: 24,
  },
  modalSectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 16,
  },
  modalItem: {
    backgroundColor: '#0a0a0a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#4CAF50',
  },
  modalIngredientName: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: 8,
  },
  modalResearchText: {
    fontSize: 14,
    color: '#ccc',
    lineHeight: 20,
  },
  modalCloseButton: {
    backgroundColor: '#4CAF50',
    borderRadius: 12,
    padding: 16,
    alignItems: 'center',
    marginTop: 16,
  },
  modalCloseButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  studyReference: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: '#0a0a0a',
    padding: 12,
    borderRadius: 8,
  },
  studyText: {
    fontSize: 13,
    color: '#4CAF50',
    marginLeft: 8,
    flex: 1,
  },
  scanAgainButton: {
    backgroundColor: '#4CAF50',
    borderRadius: 12,
    padding: 16,
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: 24,
  },
  scanAgainText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
    marginLeft: 12,
  },
  text: {
    color: '#fff',
    fontSize: 18,
  },
  scoreRingContainer: {
    width: 140,
    height: 140,
    marginBottom: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scoreRingBackground: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    borderRadius: 70,
    borderWidth: 8,
    borderColor: '#333',
  },
  scoreRingFill: {
    position: 'absolute',
    width: '100%',
    height: '100%',
    borderRadius: 70,
    borderWidth: 8,
  },
  scoreRingInner: {
    width: 100,
    height: 100,
    borderRadius: 50,
    backgroundColor: '#0c0c0c',
    justifyContent: 'center',
    alignItems: 'center',
  },
  scoreOutOf: {
    fontSize: 16,
    color: '#888',
  },
  studyLink: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: 12,
    paddingTop: 12,
    borderTopWidth: 1,
    borderTopColor: '#222',
  },
  studyLinkText: {
    fontSize: 14,
    color: '#4CAF50',
    marginLeft: 6,
    fontWeight: '600',
  },
  keyNutrients: {
    fontSize: 13,
    color: '#4CAF50',
    backgroundColor: 'rgba(76, 175, 80, 0.1)',
    padding: 8,
    borderRadius: 6,
    marginBottom: 10,
    fontWeight: '500',
  },
});
