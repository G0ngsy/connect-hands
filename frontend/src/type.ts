// frontend/src/types.ts
export interface SignData {
  word: string;
  confidence: number;
  is_detected: boolean;
  timestamp?: string;
}