import { useState, useRef, useEffect } from 'react';

// WAV Encoder Helper (PCM 16-bit Mono/Stereo)
function bufferToWav(buffer) {
  let numOfChan = buffer.numberOfChannels,
      length = buffer.length * numOfChan * 2 + 44,
      bufferArr = new ArrayBuffer(length),
      view = new DataView(bufferArr),
      channels = [], i, sample,
      offset = 0,
      pos = 0;

  // write WAVE header
  setUint32(0x46464952);                         // "RIFF"
  setUint32(length - 8);                         // file length - 8
  setUint32(0x45564157);                         // "WAVE"

  setUint32(0x20746d66);                         // "fmt " chunk
  setUint32(16);                                 // chunk length
  setUint16(1);                                  // sample format (raw PCM)
  setUint16(numOfChan);                          // channel count
  setUint32(buffer.sampleRate);                  // sample rate
  setUint32(buffer.sampleRate * 2 * numOfChan);  // byte rate
  setUint16(numOfChan * 2);                      // block align
  setUint16(16);                                 // bits per sample

  setUint32(0x61746164);                         // "data" chunk
  setUint32(length - pos - 4);                   // chunk length

  // write interleaved data
  for(i = 0; i < buffer.numberOfChannels; i++)
    channels.push(buffer.getChannelData(i));

  while(pos < length) {
    for(i = 0; i < numOfChan; i++) {             // interleave channels
      sample = Math.max(-1, Math.min(1, channels[i][offset])); // clamp
      sample = (sample < 0 ? sample * 0x8000 : sample * 0x7FFF); // scale to 16-bit signed int
      view.setInt16(pos, sample, true);          // write 16-bit sample
      pos += 2;
    }
    offset++;                                     // next sample
  }

  return new Blob([bufferArr], {type: 'audio/wav'});

  function setUint16(data) {
    view.setUint16(pos, data, true);
    pos += 2;
  }

  function setUint32(data) {
    view.setUint32(pos, data, true);
    pos += 4;
  }
}

export default function AudioDashboard({ onAnalysisStart, onAnalysisSuccess, onAnalysisError }) {
  // Recorder states
  const [isRecording, setIsRecording] = useState(false);
  const [recordingTime, setRecordingTime] = useState(0);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioBlob, setAudioBlob] = useState(null);
  
  // File upload states
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);

  // Refs for audio visualizer & media recording
  const mediaRecorderRef = useRef(null);
  const audioContextRef = useRef(null);
  const analyserRef = useRef(null);
  const animationFrameRef = useRef(null);
  const streamRef = useRef(null);
  const timerRef = useRef(null);
  const canvasRef = useRef(null);

  // Clean up visualizer & timer on unmount
  useEffect(() => {
    return () => {
      cleanupAudio();
    };
  }, []);

  const cleanupAudio = () => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
    audioContextRef.current = null;
    analyserRef.current = null;
  };

  // Start visualizer canvas drawing
  const startVisualizer = (stream) => {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const audioCtx = new AudioContext();
    const analyser = audioCtx.createAnalyser();
    const source = audioCtx.createMediaStreamSource(stream);
    
    source.connect(analyser);
    analyser.fftSize = 256;
    
    const bufferLength = analyser.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    audioContextRef.current = audioCtx;
    analyserRef.current = analyser;
    
    const canvas = canvasRef.current;
    if (!canvas) return;
    const canvasCtx = canvas.getContext('2d');
    
    const draw = () => {
      if (!analyserRef.current) return;
      animationFrameRef.current = requestAnimationFrame(draw);
      
      analyser.getByteTimeDomainData(dataArray);
      
      canvasCtx.fillStyle = 'rgba(8, 12, 20, 0.4)';
      canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
      
      canvasCtx.lineWidth = 3;
      const gradient = canvasCtx.createLinearGradient(0, 0, canvas.width, 0);
      gradient.addColorStop(0, '#6366f1'); // Indigo
      gradient.addColorStop(0.5, '#06b6d4'); // Cyan
      gradient.addColorStop(1, '#10b981'); // Emerald
      canvasCtx.strokeStyle = gradient;
      
      canvasCtx.beginPath();
      const sliceWidth = (canvas.width * 1.0) / bufferLength;
      let x = 0;
      
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        
        if (i === 0) {
          canvasCtx.moveTo(x, y);
        } else {
          canvasCtx.lineTo(x, y);
        }
        
        x += sliceWidth;
      }
      
      canvasCtx.lineTo(canvas.width, canvas.height / 2);
      canvasCtx.stroke();
    };
    
    draw();
  };

  // Toggle audio recording
  const startRecording = async () => {
    onAnalysisError(null);
    setAudioUrl(null);
    setAudioBlob(null);
    setSelectedFile(null);
    setRecordingTime(0);
    
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      
      // Start recording
      const options = { mimeType: 'audio/webm' };
      let recorder;
      try {
        recorder = new MediaRecorder(stream, options);
      } catch (e) {
        recorder = new MediaRecorder(stream);
      }
      
      const chunks = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data);
      };
      
      recorder.onstop = async () => {
        const rawBlob = new Blob(chunks, { type: recorder.mimeType });
        
        try {
          const arrayBuffer = await rawBlob.arrayBuffer();
          const AudioContext = window.AudioContext || window.webkitAudioContext;
          const tempCtx = new AudioContext();
          const decodedBuffer = await tempCtx.decodeAudioData(arrayBuffer);
          
          const wavBlob = bufferToWav(decodedBuffer);
          const url = URL.createObjectURL(wavBlob);
          setAudioBlob(wavBlob);
          setAudioUrl(url);
          tempCtx.close();
          console.log("[transcode] Audio converted to true WAV successfully.");
        } catch (e) {
          console.error("Failed to transcode to true WAV. Falling back to raw recording.", e);
          const url = URL.createObjectURL(rawBlob);
          setAudioBlob(rawBlob);
          setAudioUrl(url);
        } finally {
          cleanupAudio();
        }
      };
      
      mediaRecorderRef.current = recorder;
      recorder.start();
      setIsRecording(true);
      
      startVisualizer(stream);
      
      timerRef.current = setInterval(() => {
        setRecordingTime((prev) => prev + 1);
      }, 1000);
      
    } catch (err) {
      console.error('Microphone access denied:', err);
      onAnalysisError('Could not access the microphone. Please grant permission and try again.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  // Handle uploaded file selection
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.name.endsWith('.wav')) {
        onAnalysisError('Only .wav audio files are supported.');
        return;
      }
      onAnalysisError(null);
      setSelectedFile(file);
      setAudioBlob(file);
      setAudioUrl(URL.createObjectURL(file));
      
      const canvas = canvasRef.current;
      if (canvas) {
        const canvasCtx = canvas.getContext('2d');
        canvasCtx.fillStyle = 'rgba(8, 12, 20, 1)';
        canvasCtx.fillRect(0, 0, canvas.width, canvas.height);
      }
    }
  };

  // Trigger prediction fetch
  const handleAnalyze = async () => {
    if (!audioBlob) return;
    
    setIsAnalyzing(true);
    onAnalysisStart();
    
    const formData = new FormData();
    const fileName = selectedFile ? selectedFile.name : `recorded_speech_${Date.now()}.wav`;
    formData.append('file', audioBlob, fileName);
    
    // Explicitly POST to http://127.0.0.1:8000/predict as requested
    const apiUrl = 'http://127.0.0.1:8000/predict';
    try {
      const response = await fetch(apiUrl, {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }
      
      const data = await response.json();
      onAnalysisSuccess(data);
    } catch (err) {
      console.error('Inference request failed:', err);
      onAnalysisError('Failed to connect to the backend server. Make sure the FastAPI backend is running on port 8000.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const formatTime = (timeInSeconds) => {
    const mins = Math.floor(timeInSeconds / 60).toString().padStart(2, '0');
    const secs = (timeInSeconds % 60).toString().padStart(2, '0');
    return `${mins}:${secs}`;
  };

  return (
    <div className="recorder-section">
      <div className="visualizer-box">
        <canvas 
          ref={canvasRef} 
          className={`visualizer-canvas ${isRecording ? 'recording' : ''}`}
          width="400"
          height="140"
        />
        {!isRecording && !audioUrl && (
          <div className="visualizer-placeholder">Visualizer Idle (Start Recording)</div>
        )}
      </div>

      {isRecording && (
        <div className="recording-time">
          {formatTime(recordingTime)}
        </div>
      )}

      <div className="controls-row">
        {!isRecording ? (
          <button className="btn btn-primary" onClick={startRecording} disabled={isAnalyzing}>
            ⏺️ Start Recording
          </button>
        ) : (
          <button className="btn btn-danger pulse-recording" onClick={stopRecording}>
            ⏹️ Stop Recording
          </button>
        )}
      </div>

      <div style={{ width: '100%' }}>
        <div style={{ textAlign: 'center', color: 'var(--text-secondary)', fontSize: '0.85rem', margin: '0.75rem 0' }}>— OR —</div>
      </div>

      <label className="file-upload-zone">
        <input type="file" accept="audio/wav" onChange={handleFileChange} disabled={isRecording || isAnalyzing} />
        <div className="file-upload-text">
          📁 <span>Click to upload</span> or drag a .wav audio file
        </div>
        {selectedFile && (
          <div className="selected-file-info">
            📄 {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
          </div>
        )}
      </label>

      {audioUrl && (
        <div className="audio-player-container">
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Preview audio input:</div>
          <audio src={audioUrl} controls />
        </div>
      )}

      <button 
        className="btn btn-primary" 
        style={{ width: '100%', marginTop: '1rem', background: 'linear-gradient(135deg, var(--accent-secondary) 0%, var(--accent-primary) 100%)' }}
        onClick={handleAnalyze} 
        disabled={!audioBlob || isRecording || isAnalyzing}
      >
        {isAnalyzing ? '⚡ Analyzing Speech...' : '📊 Run Diagnostic Analysis'}
      </button>
    </div>
  );
}
