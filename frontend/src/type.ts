// 인식 결과 데이터 타입
export interface RecognitionResult {
    word: string;
    confidence: number;
    timestamp: string;
}

// 카메라 상태 타입
export type CameraStatus = 'loading' | 'active' | 'error' | 'off';