import { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  Image,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import { SafeAreaView } from 'react-native-safe-area-context';

interface HarmfulIngredient {
  name: string;
  health_risk: string;
  severity: string;
  processing_level?: string;
  study_reference: string;
}

interface BeneficialIngredient {
  name: string;
  health_benefit: string;
  processing_level?: string;
  study_reference: string;
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

  useEffect(() => {
    if (params.productData) {
      try {
        const data = JSON.parse(params.productData as string);
        setProductData(data);
      } catch (error) {
        console.error('Error parsing product data:', error);
      }
    }
  }, [params]);

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
      ? '#4CAF50'
      : analysis.overall_score >= 4
      ? '#FFA726'
      : '#FF5252';

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>
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
          <Text style={[styles.scoreValue, { color: scoreColor }]}>
            {analysis.overall_score}/10
          </Text>
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
              <Text style={styles.sectionTitle}>Harmful Ingredients</Text>
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
                <Text style={styles.ingredientDescription}>
                  {ingredient.health_risk}
                </Text>
                <View style={styles.studyReference}>
                  <Ionicons name="document-text" size={16} color="#4CAF50" />
                  <Text style={styles.studyText}>{ingredient.study_reference}</Text>
                </View>
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
                <Text style={styles.ingredientDescription}>
                  {ingredient.health_benefit}
                </Text>
                <View style={styles.studyReference}>
                  <Ionicons name="document-text" size={16} color="#4CAF50" />
                  <Text style={styles.studyText}>{ingredient.study_reference}</Text>
                </View>
              </View>
            ))}
          </View>
        )}

        <TouchableOpacity
          style={styles.scanAgainButton}
          onPress={() => router.back()}
        >
          <Ionicons name="scan" size={24} color="#fff" />
          <Text style={styles.scanAgainText}>Scan Another Product</Text>
        </TouchableOpacity>
      </ScrollView>
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
});
