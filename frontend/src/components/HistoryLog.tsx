import React from 'react';
import type { RecognitionResult } from '../type.ts';

interface Props {
    history: RecognitionResult[];
}

const HistoryLog: React.FC<Props> = ({ history }) => {
    return (
        <div className="bg-white rounded-xl shadow-lg p-6 h-full overflow-y-auto">
            <h2 className="text-xl font-bold mb-4 text-gray-800">기록</h2>
            <div className="space-y-3">
                {history.map((item, index) => (
                    <div key={index} className="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
                        <span className="font-semibold text-pink-600">{item.word}</span>
                        <span className="text-sm text-gray-400">{item.timestamp}</span>
                    </div>
                ))}
            </div>
        </div>
    );
};

export default HistoryLog;