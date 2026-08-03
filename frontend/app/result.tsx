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
  ActivityIndicator,
  Share,
  Platform,
  Alert,
  TextInput,
} from 'react-native';
import { useRouter, useLocalSearchParams } from 'expo-router';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { SafeAreaView } from 'react-native-safe-area-context';
import ConfettiCannon from 'react-native-confetti-cannon';
import * as Haptics from 'expo-haptics';
import axios from 'axios';
import { useAuth } from '../context/AuthContext';
import { askNotificationsAfterFirstScan } from '../utils/notifications';

const BACKEND_URL = process.env.EXPO_PUBLIC_BACKEND_URL || 'https://web-production-66c05.up.railway.app';

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
  is_estimate?: boolean;
  assumptions?: string[];
  refinements?: { question: string; selected?: string; options: { label: string; score: number }[] }[];
  carcinogens_found?: CarcinogenEntry[];
  chemical_breakdown?: ChemicalEntry[];
  healthier_alternatives?: AlternativeEntry[];
  shocking_facts?: ShockingFact[];
  overall_score: number;
  upf_score?: string;
  processing_category?: string;
  recommendation: string;
}

interface CarcinogenEntry {
  name: string;
  iarc_group: string;
  cancer_types: string;
  explanation: string;
  source: string;
}

interface ChemicalEntry {
  name: string;
  common_name: string;
  purpose: string;
  health_concern: string;
  banned_in: string;
}

interface AlternativeEntry {
  product_type: string;
  example_brands: string;
  why_better: string;
  score_estimate: string;
}

interface ShockingFact {
  fact: string;
  ingredient: string;
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
  const [analysisLoading, setAnalysisLoading] = useState(false);
  const [swaps, setSwaps] = useState<any[]>([]);
  const [swapsLoading, setSwapsLoading] = useState(false);
  const [expandedResearch, setExpandedResearch] = useState<{ [key: string]: boolean }>({});
  const [identifyName, setIdentifyName] = useState('');
  const [identifying, setIdentifying] = useState(false);
  const [photoBusy, setPhotoBusy] = useState(false);

  const applyNameResult = (data: any) => {
    setProductData(data);
    Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  };

  const handleQuickPick = async (name: string) => {
    if (identifying || photoBusy) return;
    setIdentifying(true);
    try {
      const res = await axios.post(
        `${BACKEND_URL}/api/analyze/name`,
        { name },
        { headers: { Authorization: `Bearer ${token}` }, timeout: 60000 }
      );
      applyNameResult(res.data);
    } catch (e: any) {
      Alert.alert('Analysis Failed', e?.response?.data?.detail || 'Could not analyze. Please try again.');
    } finally {
      setIdentifying(false);
    }
  };

  const handlePhotoScan = async () => {
    if (photoBusy || identifying) return;
    const perm = await ImagePicker.requestCameraPermissionsAsync();
    if (!perm.granted) {
      Alert.alert('Camera needed', 'Allow camera access to snap a photo of the food.');
      return;
    }
    const shot = await ImagePicker.launchCameraAsync({ quality: 0.4, base64: true, allowsEditing: false });
    if (shot.canceled || !shot.assets?.[0]?.base64) return;
    setPhotoBusy(true);
    try {
      const res = await axios.post(
        `${BACKEND_URL}/api/scan/photo`,
        { image_base64: shot.assets[0].base64 },
        { headers: { Authorization: `Bearer ${token}` }, timeout: 90000 }
      );
      if (res.data?.status === 'unclear') {
        Alert.alert("Couldn't tell what that is", 'Try a clearer photo, or use the buttons / type the name instead.');
        return;
      }
      applyNameResult(res.data);
    } catch (e: any) {
      Alert.alert('Photo scan failed', e?.response?.data?.detail || 'Could not analyze the photo. Please try again.');
    } finally {
      setPhotoBusy(false);
    }
  };

  const handleRefine = (qIdx: number, opt: { label: string; score: number }) => {
    setProductData((prev: any) => {
      if (!prev?.analysis) return prev;
      const refs = (prev.analysis.refinements || []).map((r: any, i: number) =>
        i === qIdx ? { ...r, selected: opt.label } : r);
      return { ...prev, analysis: { ...prev.analysis, overall_score: opt.score, refinements: refs } };
    });
    Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  };
  const scoreAnim = useRef(new Animated.Value(0)).current;
  const { token } = useAuth();

  useEffect(() => {
    if (productData?.analysis) {
      const t = setTimeout(() => askNotificationsAfterFirstScan(token), 2500);
      return () => clearTimeout(t);
    }
  }, [productData?.analysis]);

  useEffect(() => {
    if (productData?.analysis) {
      Animated.timing(scoreAnim, {
        toValue: productData.analysis.overall_score,
        duration: 800,
        easing: Easing.out(Easing.cubic),
        useNativeDriver: false,
      }).start();
    }
  }, [productData?.analysis]);

  const [showResearchModal, setShowResearchModal] = useState(false);
  const confettiRef = useRef<any>(null);

  useEffect(() => {
    if (params.productData) {
      try {
        const data = JSON.parse(params.productData as string);
        setProductData(data);
        
        // If analysis is complete, celebrate
        if (data.analysis) {
          if (data.analysis.overall_score >= 8) {
            confettiRef.current?.start();
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
          } else if (data.analysis.overall_score <= 3) {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
          }
        }
        
        // If needs analysis polling, start it
        if (params.needsAnalysis === 'true' && params.barcode) {
          setAnalysisLoading(true);
          pollForAnalysis(params.barcode as string);
        }
      } catch (error) {
        console.error('Error parsing product data:', error);
      }
    }
  }, []);

  const pollForAnalysis = async (barcode: string) => {
    const maxAttempts = 20; // Poll for up to 60 seconds (20 * 3s)
    let attempts = 0;
    
    const poll = async () => {
      if (attempts >= maxAttempts) {
        setAnalysisLoading(false);
        return;
      }
      attempts++;
      
      try {
        const response = await axios.get(
          `${BACKEND_URL}/api/scan/status/${barcode}`,
          { headers: { Authorization: `Bearer ${token}` }, timeout: 10000 }
        );
        
        if (response.data.status === 'complete' && response.data.analysis) {
          setProductData(response.data);
          setAnalysisLoading(false);
          
          // Celebrate/warn based on score
          if (response.data.analysis.overall_score >= 8) {
            confettiRef.current?.start();
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
          } else if (response.data.analysis.overall_score <= 3) {
            Haptics.notificationAsync(Haptics.NotificationFeedbackType.Warning);
          }

          // Update gamification streak (non-blocking)
          try {
            await axios.post(
              `${BACKEND_URL}/api/gamification/update-streak`,
              {},
              { headers: { Authorization: `Bearer ${token}` } }
            );
          } catch (e) {
            console.log('Gamification update failed');
          }
          return;
        }
        
        // AI analysis failed on backend
        if (response.data.status === 'error') {
          setAnalysisLoading(false);
          return;
        }
        
        // Still analyzing, poll again in 3 seconds
        setTimeout(poll, 3000);
      } catch (error) {
        console.error('Poll error:', error);
        setTimeout(poll, 3000);
      }
    };
    
    // Start polling after 3 seconds
    setTimeout(poll, 3000);
  };


  // Fetch healthier swaps after analysis loads
  useEffect(() => {
    const barcode = params.barcode as string;
    const currentAnalysis = productData?.analysis;
    if (currentAnalysis && barcode && token && currentAnalysis.overall_score <= 7) {
      setSwapsLoading(true);
      axios.post(`${BACKEND_URL}/api/scan/swaps`, { barcode }, {
        headers: { Authorization: `Bearer ${token}` },
        timeout: 15000,
      })
      .then(res => {
        if (res.data?.swaps?.length > 0) {
          setSwaps(res.data.swaps);
        }
      })
      .catch(() => {}) // Silently fail — swaps are optional
      .finally(() => setSwapsLoading(false));
    }
  }, [productData?.analysis]);

  const getNovaColor = (level?: string) => {
    if (!level) return '#2196F3';
    const l = level.toLowerCase();
    if (l.includes('4')) return '#FF5252';      // Red — Ultra-Processed
    if (l.includes('3')) return '#FFA726';      // Amber — Processed
    if (l.includes('2')) return '#8BC34A';      // Yellow-Green — Culinary Ingredients
    if (l.includes('1')) return '#00E676';      // Green — Whole Food
    return '#2196F3';
  };

  const toggleResearch = (ingredientName: string) => {
    setExpandedResearch(prev => ({
      ...prev,
      [ingredientName]: !prev[ingredientName]
    }));
  };

  const handleShare = async () => {
    if (!productData || !analysis) return;
    const score = analysis.overall_score;
    const category = analysis.processing_category || 'Unknown';
    const harmfulCount = analysis.harmful_ingredients?.length || 0;
    const carcinogenCount = analysis.carcinogens_found?.length || 0;

    let headline = '';
    if (score <= 3) headline = `${product_name} scored ${score}/10 — avoid this!`;
    else if (score <= 6) headline = `${product_name} scored ${score}/10 — could be better.`;
    else headline = `${product_name} scored ${score}/10 — a healthy choice!`;

    let details = `Category: ${category}`;
    if (harmfulCount > 0) details += `\n${harmfulCount} harmful ingredient${harmfulCount > 1 ? 's' : ''} found`;
    if (carcinogenCount > 0) details += `\n${carcinogenCount} carcinogen${carcinogenCount > 1 ? 's' : ''} detected!`;
    if (analysis.shocking_facts?.length) details += `\n\nDid you know? ${analysis.shocking_facts[0].fact}`;

    const playStoreUrl = 'https://play.google.com/store/apps/details?id=com.youarewhatyoueat.app';
    const appStoreUrl = 'https://apps.apple.com/app/you-are-what-you-eat/id6743126498';
    const downloadLink = Platform.OS === 'ios' ? appStoreUrl : playStoreUrl;
    const otherLink = Platform.OS === 'ios' ? playStoreUrl : appStoreUrl;

    try {
      await Share.share({
        message: `${headline}\n${details}\n\nScan your food with You Are What You Eat:\n${downloadLink}\n\nAlso available: ${otherLink}`,
      });
    } catch (e) {
      console.log('Share error:', e);
    }
  };

  if (!productData) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Loading...</Text>
      </View>
    );
  }

  const { product_name, brands, image_url, analysis } = productData;
  const isUnidentified = !!analysis && (product_name || '').toLowerCase().includes('unknown');

  const handleIdentify = async () => {
    const name = identifyName.trim();
    if (!name || name.length < 3) return;
    setIdentifying(true);
    try {
      const res = await axios.post(
        `${BACKEND_URL}/api/scan/identify`,
        { barcode: params.barcode, product_name: name },
        { headers: { Authorization: `Bearer ${token}` }, timeout: 60000 }
      );
      setProductData(res.data);
      setIdentifyName('');
      Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
    } catch (e: any) {
      Alert.alert('Analysis Failed', e?.response?.data?.detail || 'Could not analyze this product. Please try again.');
    } finally {
      setIdentifying(false);
    }
  };

  const scoreColor = !analysis ? '#4CAF50' :
    analysis.overall_score >= 7
      ? '#00E676'
      : analysis.overall_score >= 4
      ? '#FFD54F'
      : '#FF5252';
  
  const scoreGradient = !analysis ? ['#4CAF50', '#388E3C'] :
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

        {analysis && (
          <View style={styles.actionRow}>
            <TouchableOpacity style={styles.shareButton} onPress={handleShare} data-testid="share-result-btn">
              <Ionicons name="share-social" size={22} color="#fff" />
              <Text style={styles.shareButtonText}>Share Result</Text>
            </TouchableOpacity>
            <TouchableOpacity 
              style={styles.rescanButton} 
              onPress={async () => {
                try {
                  setAnalysisLoading(true);
                  const bc = params.barcode as string;
                  const res = await axios.post(`${BACKEND_URL}/api/scan/rescan`, { barcode: bc }, {
                    headers: { Authorization: `Bearer ${token}` }
                  });
                  if (res.data?.analysis) {
                    setProductData(prev => prev ? { ...prev, analysis: res.data.analysis } : prev);
                  }
                } catch (e: any) {
                  console.warn('Rescan error:', e);
                  Alert.alert('Re-scan Failed', 'Could not re-analyse this product. Please try again.');
                } finally {
                  setAnalysisLoading(false);
                }
              }}
              disabled={analysisLoading}
              data-testid="rescan-btn"
            >
              <Ionicons name="refresh" size={22} color="#4CAF50" />
              <Text style={styles.rescanButtonText}>Re-scan</Text>
            </TouchableOpacity>
          </View>
        )}

        {analysisLoading && !analysis ? (
          <View style={[styles.scoreCard, { borderColor: '#4CAF50' }]}>
            <ActivityIndicator size="large" color="#4CAF50" />
            <Text style={[styles.scoreLabel, { marginTop: 16 }]}>Analyzing ingredients...</Text>
            <Text style={[styles.recommendation, { marginTop: 8, fontSize: 14 }]}>
              Our AI is reviewing this product. This usually takes 10-15 seconds.
            </Text>
          </View>
        ) : analysis ? (
          <>
        {!isUnidentified && (
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
            <View style={[styles.processingBadge, {
              backgroundColor:
                analysis.processing_category.toLowerCase().includes('ultra')
                  ? '#FF5252'
                  : analysis.processing_category.toLowerCase().includes('processed') && !analysis.processing_category.toLowerCase().includes('minimally')
                  ? '#FFA726'
                  : analysis.processing_category.toLowerCase().includes('minimally')
                  ? '#8BC34A'
                  : '#00E676',
            }]}>
              <Text style={styles.processingText}>{analysis.processing_category}</Text>
            </View>
          )}
          {analysis.upf_score && (
            <Text style={styles.upfScore}>UPF Content: {analysis.upf_score}</Text>
          )}
          <Text style={styles.recommendation}>{analysis.recommendation}</Text>
        </View>
        )}

        {isUnidentified && (
          <View style={styles.identifyCard} data-testid="identify-product-card">
            <View style={styles.identifyHeader}>
              <Ionicons name="help-circle" size={22} color="#FFD54F" />
              <Text style={styles.identifyTitle}>What product is this?</Text>
            </View>
            <Text style={styles.identifyHint}>
              {(product_name || '').toLowerCase().includes('store label')
                ? "No barcode? Usually a great sign 🥦 — the healthiest food in the store doesn't need a label. That was a store's own tag for a weighed, deli or bakery item. Tell us what it is:"
                : "This barcode isn't in any food database yet. Type the product name and our AI will score it in ~10 seconds — you'll also unlock it for every future scanner! 🎉"}
            </Text>
            <View style={styles.quickPickRow}>
              {[
                { label: '🥖 Bakery', name: 'fresh bakery item (bread, roll or pastry)' },
                { label: '🥩 Deli meat', name: 'deli counter sliced processed meat (ham, salami)' },
                { label: '🧀 Cheese', name: 'cheese from the cheese counter' },
                { label: '🥗 Fruit & veg', name: 'fresh fruit and vegetables' },
              ].map((p) => (
                <TouchableOpacity
                  key={p.label}
                  style={styles.quickPickChip}
                  disabled={identifying || photoBusy}
                  onPress={() => handleQuickPick(p.name)}
                  data-testid={`quick-pick-${p.label.slice(3).toLowerCase().replace(/[^a-z]+/g, '-')}`}
                >
                  <Text style={styles.quickPickText}>{p.label}</Text>
                </TouchableOpacity>
              ))}
            </View>
            <TouchableOpacity
              style={styles.photoButton}
              disabled={photoBusy || identifying}
              onPress={handlePhotoScan}
              data-testid="photo-scan-button"
            >
              {photoBusy
                ? <ActivityIndicator color="#FFD54F" />
                : <>
                    <Ionicons name="camera" size={20} color="#FFD54F" />
                    <Text style={styles.photoButtonText}>Snap a photo — AI identifies it</Text>
                  </>}
            </TouchableOpacity>
            <TextInput
              style={styles.identifyInput}
              placeholder="e.g. Tesco Chocolate Digestives"
              placeholderTextColor="#777"
              value={identifyName}
              onChangeText={setIdentifyName}
              editable={!identifying}
              autoFocus
              returnKeyType="done"
              onSubmitEditing={handleIdentify}
              data-testid="identify-product-input"
            />
            <TouchableOpacity
              style={[styles.identifyButton, (!identifyName.trim() || identifying) && { opacity: 0.5 }]}
              disabled={!identifyName.trim() || identifying}
              onPress={handleIdentify}
              data-testid="identify-product-submit"
            >
              {identifying
                ? <ActivityIndicator color="#000" />
                : <Text style={styles.identifyButtonText}>Analyze it</Text>}
            </TouchableOpacity>
          </View>
        )}

        {/* Photo/name estimate: assumptions + refinement chips */}
        {analysis.is_estimate && (
          <View style={styles.estimateCard} data-testid="estimate-card">
            <View style={styles.identifyHeader}>
              <Ionicons name="analytics" size={20} color="#FFD54F" />
              <Text style={styles.identifyTitle}>Typical score — help us refine it</Text>
            </View>
            {(analysis.assumptions || []).length > 0 && (
              <Text style={styles.assumptionText}>
                {(analysis.assumptions || []).map((a: string) => `• ${a}`).join('\n')}
              </Text>
            )}
            {(analysis.refinements || []).map((r: any, qi: number) => (
              <View key={qi} style={{ marginTop: 10 }}>
                <Text style={styles.refineQuestion}>{r.question}</Text>
                <View style={styles.quickPickRow}>
                  {(r.options || []).map((o: any) => (
                    <TouchableOpacity
                      key={o.label}
                      style={[styles.quickPickChip, r.selected === o.label && styles.quickPickChipActive]}
                      onPress={() => handleRefine(qi, o)}
                      data-testid="refine-option"
                    >
                      <Text style={[styles.quickPickText, r.selected === o.label && { color: '#000' }]}>
                        {o.label} · {o.score}/10
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </View>
            ))}
          </View>
        )}

        {/* Shocking Facts - "Did You Know?" Section */}
        {analysis.shocking_facts && analysis.shocking_facts.length > 0 && (
          <View style={styles.shockingSection}>
            <View style={styles.shockingHeader}>
              <Ionicons name="alert-circle" size={22} color="#FFD600" />
              <Text style={styles.shockingTitle}>Did You Know?</Text>
            </View>
            {analysis.shocking_facts.map((item, index) => (
              <View key={index} style={styles.shockingCard}>
                <Text style={styles.shockingFact}>{item.fact}</Text>
                <Text style={styles.shockingIngredient}>{item.ingredient}</Text>
              </View>
            ))}
          </View>
        )}

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
                  <Text style={[styles.processingLevel, { color: getNovaColor(ingredient.processing_level) }]}>{ingredient.processing_level}</Text>
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
                  <Text style={[styles.processingLevel, { color: getNovaColor(ingredient.processing_level) }]}>{ingredient.processing_level}</Text>
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

        {/* Carcinogens Warning Section */}
        {analysis.carcinogens_found && analysis.carcinogens_found.length > 0 && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Ionicons name="skull" size={24} color="#FF1744" />
              <Text style={styles.sectionTitle}>Carcinogens Detected</Text>
            </View>
            {analysis.carcinogens_found.map((item, index) => (
              <View key={index} style={styles.carcinogenCard}>
                <View style={styles.ingredientHeader}>
                  <Text style={styles.ingredientName}>{item.name}</Text>
                  <View style={styles.iarcBadge}>
                    <Text style={styles.iarcText}>{item.iarc_group}</Text>
                  </View>
                </View>
                <Text style={styles.cancerTypes}>Linked to: {item.cancer_types}</Text>
                <Text style={styles.healthImpact}>{item.explanation}</Text>
                <Text style={styles.carcinogenSource}>{item.source}</Text>
              </View>
            ))}
          </View>
        )}

        {/* Chemical Breakdown Section */}
        {analysis.chemical_breakdown && analysis.chemical_breakdown.length > 0 && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Ionicons name="flask" size={24} color="#FF9100" />
              <Text style={styles.sectionTitle}>Chemical Breakdown</Text>
            </View>
            {analysis.chemical_breakdown.map((chem, index) => (
              <View key={index} style={styles.chemicalCard}>
                <View style={styles.chemicalHeader}>
                  <Text style={styles.chemicalName}>{chem.name}</Text>
                  <Text style={styles.chemicalCommon}>{chem.common_name}</Text>
                </View>
                <Text style={styles.chemicalPurpose}>Used for: {chem.purpose}</Text>
                <Text style={styles.chemicalConcern}>{chem.health_concern}</Text>
                {chem.banned_in ? (
                  <View style={styles.bannedBadge}>
                    <Ionicons name="ban" size={14} color="#FF5252" />
                    <Text style={styles.bannedText}>Banned in: {chem.banned_in}</Text>
                  </View>
                ) : null}
              </View>
            ))}
          </View>
        )}

        {/* Healthier Alternatives Section */}
        {analysis.healthier_alternatives && analysis.healthier_alternatives.length > 0 && (
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Ionicons name="swap-horizontal" size={24} color="#00E676" />
              <Text style={styles.sectionTitle}>Healthier Alternatives</Text>
            </View>
            {analysis.healthier_alternatives.map((alt, index) => (
              <View key={index} style={styles.alternativeCard}>
                <View style={styles.altHeader}>
                  <Ionicons name="leaf" size={20} color="#00E676" />
                  <Text style={styles.altTitle}>{alt.product_type}</Text>
                  <View style={styles.altScoreBadge}>
                    <Text style={styles.altScoreText}>{alt.score_estimate}</Text>
                  </View>
                </View>
                {alt.example_brands ? (
                  <Text style={styles.altBrands}>Try: {alt.example_brands}</Text>
                ) : null}
                <Text style={styles.altReason}>{alt.why_better}</Text>
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

        {/* Healthier Swaps — loads async, never blocks main result */}
        {swaps.length > 0 && (
          <View style={styles.swapsSection}>
            <View style={styles.swapsHeader}>
              <Ionicons name="swap-horizontal" size={22} color="#00e676" />
              <Text style={styles.swapsTitle}>Healthier Swaps</Text>
            </View>
            {swaps.map((swap: any, idx: number) => (
              <View key={idx} style={styles.swapCard}>
                {swap.image_url ? (
                  <Image source={{ uri: swap.image_url }} style={styles.swapImage} />
                ) : (
                  <View style={[styles.swapImage, styles.swapImagePlaceholder]}>
                    <Ionicons name="leaf" size={20} color="#00e676" />
                  </View>
                )}
                <View style={styles.swapInfo}>
                  <Text style={styles.swapName} numberOfLines={2}>{swap.product_name}</Text>
                  {swap.brands ? <Text style={styles.swapBrand}>{swap.brands}</Text> : null}
                  {swap.why_better ? <Text style={styles.swapWhy}>{swap.why_better}</Text> : null}
                  {swap.ingredient_count ? (
                    <Text style={styles.swapDetail}>{swap.ingredient_count} ingredients</Text>
                  ) : null}
                </View>
                {swap.score ? (
                  <View style={styles.swapScore}>
                    <Text style={styles.swapScoreText}>{swap.score}</Text>
                    <Text style={styles.swapScoreLabel}>/10</Text>
                  </View>
                ) : null}
              </View>
            ))}
          </View>
        )}
        {swapsLoading && (
          <View style={styles.swapsLoading}>
            <ActivityIndicator size="small" color="#00e676" />
            <Text style={styles.swapsLoadingText}>Finding healthier alternatives...</Text>
          </View>
        )}

        {/* Static Sources & References — always visible for Apple Review compliance */}
        <View style={styles.sourcesSection}>
          <View style={styles.sourcesHeader}>
            <Ionicons name="document-text-outline" size={20} color="#4CAF50" />
            <Text style={styles.sourcesTitle}>Sources & References</Text>
          </View>
          <Text style={styles.sourcesIntro}>
            Health assessments in this app are based on the following peer-reviewed sources and classification systems:
          </Text>
          <TouchableOpacity onPress={() => Linking.openURL('https://pubmed.ncbi.nlm.nih.gov/31105044/')}>
            <Text style={styles.sourceLink}>Monteiro CA et al. (2019) "Ultra-processed foods: what they are and how to identify them." Public Health Nutrition, 20(5), 936-941. [NOVA Classification]</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => Linking.openURL('https://monographs.iarc.who.int/list-of-classifications')}>
            <Text style={styles.sourceLink}>IARC Monographs on the Identification of Carcinogenic Hazards to Humans — World Health Organization</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => Linking.openURL('https://pubmed.ncbi.nlm.nih.gov/36543367/')}>
            <Text style={styles.sourceLink}>Lane MM et al. (2024) "Ultra-processed food exposure and adverse health outcomes." BMJ, 383, e077310</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => Linking.openURL('https://www.who.int/news-room/fact-sheets/detail/healthy-diet')}>
            <Text style={styles.sourceLink}>World Health Organization — Healthy Diet Fact Sheet</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => Linking.openURL('https://pubmed.ncbi.nlm.nih.gov/30742202/')}>
            <Text style={styles.sourceLink}>Srour B et al. (2019) "Ultra-processed food intake and risk of cardiovascular disease." BMJ, 365, l1451</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => Linking.openURL('https://www.efsa.europa.eu/en/topics/topic/food-additives')}>
            <Text style={styles.sourceLink}>European Food Safety Authority (EFSA) — Food Additives Database</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => Linking.openURL('https://www.hsph.harvard.edu/nutritionsource/')}>
            <Text style={styles.sourceLink}>Harvard T.H. Chan School of Public Health — The Nutrition Source</Text>
          </TouchableOpacity>
          <Text style={styles.sourcesDisclaimer}>
            Individual ingredient research summaries and study links are available in each ingredient's expandable "Research" section above. This app provides educational information only — consult a healthcare professional for personal medical advice.
          </Text>
        </View>
          </>
        ) : null}

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
  actionRow: {
    flexDirection: 'row',
    marginBottom: 20,
  },
  shareButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2196F3',
    borderRadius: 24,
    paddingHorizontal: 20,
    paddingVertical: 10,
  },
  shareButtonText: {
    color: '#fff',
    fontSize: 15,
    fontWeight: '600',
    marginLeft: 8,
  },
  rescanButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    paddingVertical: 12,
    paddingHorizontal: 20,
    borderWidth: 1,
    borderColor: '#4CAF50',
  },
  rescanButtonText: {
    color: '#4CAF50',
    fontSize: 15,
    fontWeight: '600',
    marginLeft: 8,
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
  quickPickRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
    marginTop: 10,
  },
  quickPickChip: {
    backgroundColor: 'rgba(255,213,79,0.12)',
    borderWidth: 1,
    borderColor: 'rgba(255,213,79,0.4)',
    borderRadius: 20,
    paddingHorizontal: 14,
    paddingVertical: 8,
  },
  quickPickChipActive: {
    backgroundColor: '#FFD54F',
  },
  quickPickText: {
    color: '#FFD54F',
    fontSize: 13,
    fontWeight: '600',
  },
  photoButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    borderWidth: 1,
    borderColor: 'rgba(255,213,79,0.5)',
    borderRadius: 12,
    paddingVertical: 12,
    marginTop: 12,
  },
  photoButtonText: {
    color: '#FFD54F',
    fontSize: 14,
    fontWeight: '700',
  },
  estimateCard: {
    backgroundColor: 'rgba(255,213,79,0.06)',
    borderWidth: 1,
    borderColor: 'rgba(255,213,79,0.25)',
    borderRadius: 16,
    padding: 16,
    marginBottom: 20,
  },
  assumptionText: {
    color: '#AAA',
    fontSize: 12,
    lineHeight: 18,
    marginTop: 6,
  },
  refineQuestion: {
    color: '#FFF',
    fontSize: 13,
    fontWeight: '600',
  },
  identifyCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#FFD54F',
    padding: 16,
    marginHorizontal: 16,
    marginBottom: 16,
  },
  identifyHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 8,
  },
  identifyTitle: {
    color: '#FFD54F',
    fontSize: 16,
    fontWeight: 'bold',
  },
  identifyHint: {
    color: '#bbb',
    fontSize: 13,
    marginBottom: 12,
    lineHeight: 18,
  },
  identifyInput: {
    backgroundColor: '#000',
    borderWidth: 1,
    borderColor: '#333',
    borderRadius: 10,
    color: '#fff',
    paddingHorizontal: 14,
    paddingVertical: 12,
    fontSize: 15,
    marginBottom: 10,
  },
  identifyButton: {
    backgroundColor: '#FFD54F',
    borderRadius: 10,
    paddingVertical: 12,
    alignItems: 'center',
  },
  identifyButtonText: {
    color: '#000',
    fontSize: 15,
    fontWeight: 'bold',
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
  carcinogenCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#FF1744',
  },
  iarcBadge: {
    backgroundColor: '#FF1744',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 10,
  },
  iarcText: {
    fontSize: 11,
    fontWeight: 'bold',
    color: '#fff',
  },
  cancerTypes: {
    fontSize: 13,
    color: '#FF8A80',
    marginBottom: 8,
    fontStyle: 'italic',
  },
  carcinogenSource: {
    fontSize: 12,
    color: '#666',
    marginTop: 8,
    fontStyle: 'italic',
  },
  chemicalCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#FF9100',
  },
  chemicalHeader: {
    marginBottom: 8,
  },
  chemicalName: {
    fontSize: 17,
    fontWeight: 'bold',
    color: '#FF9100',
  },
  chemicalCommon: {
    fontSize: 14,
    color: '#aaa',
    marginTop: 2,
  },
  chemicalPurpose: {
    fontSize: 13,
    color: '#888',
    marginBottom: 6,
  },
  chemicalConcern: {
    fontSize: 14,
    color: '#fff',
    lineHeight: 20,
    marginBottom: 8,
  },
  bannedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(255, 82, 82, 0.15)',
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 8,
    alignSelf: 'flex-start',
  },
  bannedText: {
    fontSize: 12,
    color: '#FF5252',
    marginLeft: 6,
    fontWeight: '600',
  },
  alternativeCard: {
    backgroundColor: '#1a1a1a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 3,
    borderLeftColor: '#00E676',
  },
  altHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  altTitle: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#fff',
    marginLeft: 8,
    flex: 1,
  },
  altScoreBadge: {
    backgroundColor: 'rgba(0, 230, 118, 0.2)',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 10,
  },
  altScoreText: {
    fontSize: 13,
    fontWeight: 'bold',
    color: '#00E676',
  },
  altBrands: {
    fontSize: 14,
    color: '#00E676',
    marginBottom: 6,
    fontWeight: '500',
  },
  altReason: {
    fontSize: 14,
    color: '#ccc',
    lineHeight: 20,
  },
  shockingSection: {
    marginBottom: 20,
    backgroundColor: 'rgba(255, 214, 0, 0.08)',
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: 'rgba(255, 214, 0, 0.25)',
  },
  shockingHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
  },
  shockingTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#FFD600',
    marginLeft: 8,
  },
  shockingCard: {
    backgroundColor: 'rgba(0, 0, 0, 0.3)',
    borderRadius: 12,
    padding: 14,
    marginBottom: 10,
    borderLeftWidth: 3,
    borderLeftColor: '#FFD600',
  },
  shockingFact: {
    fontSize: 15,
    color: '#fff',
    lineHeight: 22,
    fontWeight: '500',
  },
  shockingIngredient: {
    fontSize: 12,
    color: '#FFD600',
    marginTop: 8,
    fontWeight: '600',
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  swapsSection: {
    backgroundColor: '#0d1a0d',
    borderRadius: 16,
    padding: 20,
    marginTop: 20,
    borderWidth: 1,
    borderColor: '#00e676',
  },
  swapsHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  swapsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#00e676',
    marginLeft: 8,
  },
  swapCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#111',
    borderRadius: 12,
    padding: 12,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#1a1a1a',
  },
  swapImage: {
    width: 48,
    height: 48,
    borderRadius: 8,
    backgroundColor: '#1a1a1a',
  },
  swapImagePlaceholder: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  swapInfo: {
    flex: 1,
    marginLeft: 12,
  },
  swapName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#fff',
  },
  swapBrand: {
    fontSize: 12,
    color: '#888',
    marginTop: 2,
  },
  swapWhy: {
    fontSize: 12,
    color: '#aaa',
    marginTop: 4,
    fontStyle: 'italic',
  },
  swapDetail: {
    fontSize: 11,
    color: '#666',
    marginTop: 2,
  },
  swapScore: {
    alignItems: 'center',
    marginLeft: 12,
  },
  swapScoreText: {
    fontSize: 22,
    fontWeight: '900',
    color: '#00e676',
  },
  swapScoreLabel: {
    fontSize: 10,
    color: '#666',
  },
  swapsLoading: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    padding: 16,
    marginTop: 16,
  },
  swapsLoadingText: {
    color: '#666',
    fontSize: 13,
    marginLeft: 8,
  },
  sourcesSection: {
    backgroundColor: '#1a1a1a',
    borderRadius: 16,
    padding: 20,
    marginTop: 20,
    borderWidth: 2,
    borderColor: '#4CAF50',
  },
  sourcesHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
    backgroundColor: 'rgba(76, 175, 80, 0.15)',
    padding: 12,
    borderRadius: 10,
  },
  sourcesTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#4CAF50',
    marginLeft: 8,
  },
  sourcesIntro: {
    fontSize: 13,
    color: '#aaa',
    lineHeight: 18,
    marginBottom: 12,
  },
  sourceLink: {
    fontSize: 13,
    color: '#4CAF50',
    lineHeight: 19,
    marginBottom: 10,
    paddingLeft: 8,
    borderLeftWidth: 2,
    borderLeftColor: '#4CAF50',
  },
  sourcesDisclaimer: {
    fontSize: 11,
    color: '#666',
    lineHeight: 16,
    marginTop: 12,
    fontStyle: 'italic',
    borderTopWidth: 1,
    borderTopColor: '#222',
    paddingTop: 12,
  },
});

