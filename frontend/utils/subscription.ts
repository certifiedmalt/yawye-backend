import * as Localization from 'expo-localization';

// Currency mapping by country/region
const CURRENCY_MAP: { [key: string]: { code: string; symbol: string; price: { monthly: number; yearly: number } } } = {
  // United States
  US: { code: 'USD', symbol: '$', price: { monthly: 4.99, yearly: 39.99 } },
  
  // United Kingdom
  GB: { code: 'GBP', symbol: '£', price: { monthly: 3.99, yearly: 32.99 } },
  
  // Eurozone
  AT: { code: 'EUR', symbol: '€', price: { monthly: 4.49, yearly: 36.99 } },
  BE: { code: 'EUR', symbol: '€', price: { monthly: 4.49, yearly: 36.99 } },
  DE: { code: 'EUR', symbol: '€', price: { monthly: 4.49, yearly: 36.99 } },
  ES: { code: 'EUR', symbol: '€', price: { monthly: 4.49, yearly: 36.99 } },
  FR: { code: 'EUR', symbol: '€', price: { monthly: 4.49, yearly: 36.99 } },
  GR: { code: 'EUR', symbol: '€', price: { monthly: 4.49, yearly: 36.99 } },
  IE: { code: 'EUR', symbol: '€', price: { monthly: 4.49, yearly: 36.99 } },
  IT: { code: 'EUR', symbol: '€', price: { monthly: 4.49, yearly: 36.99 } },
  NL: { code: 'EUR', symbol: '€', price: { monthly: 4.49, yearly: 36.99 } },
  PT: { code: 'EUR', symbol: '€', price: { monthly: 4.49, yearly: 36.99 } },
  
  // Other European
  CH: { code: 'CHF', symbol: 'CHF', price: { monthly: 4.99, yearly: 39.99 } },
  NO: { code: 'NOK', symbol: 'kr', price: { monthly: 49, yearly: 399 } },
  SE: { code: 'SEK', symbol: 'kr', price: { monthly: 49, yearly: 399 } },
  DK: { code: 'DKK', symbol: 'kr', price: { monthly: 35, yearly: 285 } },
  
  // Americas
  CA: { code: 'CAD', symbol: 'C$', price: { monthly: 6.49, yearly: 52.99 } },
  MX: { code: 'MXN', symbol: '$', price: { monthly: 99, yearly: 799 } },
  BR: { code: 'BRL', symbol: 'R$', price: { monthly: 24.99, yearly: 199.99 } },
  AR: { code: 'ARS', symbol: '$', price: { monthly: 999, yearly: 7999 } },
  
  // Asia Pacific
  AU: { code: 'AUD', symbol: 'A$', price: { monthly: 7.49, yearly: 59.99 } },
  NZ: { code: 'NZD', symbol: 'NZ$', price: { monthly: 7.99, yearly: 64.99 } },
  JP: { code: 'JPY', symbol: '¥', price: { monthly: 699, yearly: 5699 } },
  IN: { code: 'INR', symbol: '₹', price: { monthly: 399, yearly: 3199 } },
  SG: { code: 'SGD', symbol: 'S$', price: { monthly: 6.49, yearly: 52.99 } },
  
  // Default fallback
  DEFAULT: { code: 'USD', symbol: '$', price: { monthly: 4.99, yearly: 39.99 } },
};

export function getCurrencyForRegion(): { code: string; symbol: string; price: { monthly: number; yearly: number } } {
  const region = Localization.region || 'US';
  return CURRENCY_MAP[region] || CURRENCY_MAP.DEFAULT;
}

export function formatPrice(amount: number, currencyCode: string, currencySymbol: string): string {
  // Format based on currency
  if (currencyCode === 'JPY' || currencyCode === 'KRW') {
    // No decimals for Yen and Won
    return `${currencySymbol}${Math.round(amount).toLocaleString()}`;
  }
  
  // Standard formatting with 2 decimals
  return `${currencySymbol}${amount.toFixed(2)}`;
}

export const SUBSCRIPTION_PRODUCTS = {
  monthly: {
    id: 'com.youarewhatyoueat.premium.monthly',
    title: 'Premium Monthly',
  },
  yearly: {
    id: 'com.youarewhatyoueat.premium.yearly',
    title: 'Premium Yearly',
  },
};
