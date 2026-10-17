import { useState } from 'react';
import './App.css';
import AudioDashboard from './AudioDashboard';

function App() {
  // State managed globally for rendering results
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  const handleAnalysisStart = () => {
    setIsAnalyzing(true);
    setError(null);
    setResults(null);
  };

  const handleAnalysisSuccess = (data) => {
    setResults(data);
    setIsAnalyzing(false);
  };

  const handleAnalysisError = (errMsg) => {
    setError(errMsg);
    setIsAnalyzing(false);
  };

  // Supports both exact label strings from backend outputs
  const isAD = results?.prediction === 'AD' || results?.prediction === "Alzheimer's Disease";

  return (
    <div className="dashboard-container">
      {/* Header */}
      <header className="header">
        <h1>Alzheimer's Multi-Modal Diagnosis</h1>
        <p>AI-Powered Clinical Analysis combining Acoustic & Linguistic Features</p>
      </header>

      {error && (
        <div className="error-message">
          <strong>⚠️ Error:</strong> {error}
        </div>
      )}

      {/* Main Grid */}
      <main className="dashboard-grid">
        
        {/* Left Column: Recording & Input Component */}
        <section className="card">
          <h2>🎙️ Speech & Audio Input</h2>
          <AudioDashboard 
            onAnalysisStart={handleAnalysisStart}
            onAnalysisSuccess={handleAnalysisSuccess}
            onAnalysisError={handleAnalysisError}
          />
        </section>

        {/* Right Column: Diagnostic Dashboard Outputs */}
        <section className="card">
          <h2>📈 Diagnostic Results</h2>
          
          {!results ? (
            <div style={{ textAlign: 'center', padding: '4rem 1rem', color: 'var(--text-secondary)' }}>
              {isAnalyzing ? (
                <div>
                  <div style={{ fontSize: '1.2rem', marginBottom: '0.5rem', color: 'var(--text-primary)' }}>Evaluating Multi-modal Pipeline...</div>
                  <p style={{ fontSize: '0.9rem' }}>Passing WavLM Conformer & ClinicalBERT Gated Transformer features through Cross-Modal Attention fusion.</p>
                </div>
              ) : (
                <div>
                  <div style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>📋</div>
                  <p>Awaiting audio input analysis. Upload a WAV file or record your speech to trigger the diagnostics model.</p>
                </div>
              )}
            </div>
          ) : (
            <div className="results-section">
              
              {/* Classification Outcome */}
              <div className="results-header-block">
                <div>
                  <div className="diagnosis-label">Model Classification</div>
                  <div className={`diagnosis-value ${isAD ? 'ad' : 'hc'}`}>
                    {results.prediction}
                  </div>
                </div>
                <span className={`badge ${isAD ? 'badge-ad' : 'badge-hc'}`}>
                  {isAD ? 'High Risk' : 'Healthy'}
                </span>
              </div>

              {/* Metrics Grid */}
              <div className="metrics-grid">
                
                {/* Risk Score */}
                <div className="metric-card">
                  <span className="metric-title">AD Risk Probability</span>
                  <div className="metric-value-row">
                    <span className="metric-number">{(results.risk_score * 100).toFixed(2)}</span>
                    <span className="metric-unit">%</span>
                  </div>
                  <div className="progress-bar-container">
                    <div 
                      className={`progress-bar ${isAD ? 'progress-ad' : 'progress-hc'}`}
                      style={{ width: `${results.risk_score * 100}%` }}
                    />
                  </div>
                </div>

                {/* Confidence */}
                <div className="metric-card">
                  <span className="metric-title">Model Confidence</span>
                  <div className="metric-value-row">
                    <span className="metric-number">{(results.confidence * 100).toFixed(2)}</span>
                    <span className="metric-unit">%</span>
                  </div>
                  <div className="progress-bar-container">
                    <div 
                      className={`progress-bar ${isAD ? 'progress-ad' : 'progress-hc'}`}
                      style={{ width: `${results.confidence * 100}%` }}
                    />
                  </div>
                </div>

              </div>

              {/* Transcript Display */}
              <div className="transcript-card">
                <div style={{ fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>
                  Acoustic Speech Transcript (ClinicalBERT Input)
                </div>
                <p className="transcript-text">"{results.transcript}"</p>
              </div>

            </div>
          )}
        </section>

        {/* Bottom Panel: Model Architecture */}
        <section className="card architecture-card">
          <h2>🧠 Upgraded Tri-Modal Neural Network Architecture</h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '0.25rem' }}>
            This diagram details the deep learning pipeline that runs forward pass evaluations inside the FastAPI backend.
          </p>
          <div className="architecture-flow">
            <div className="flow-node">
              <strong>🎙️ Audio Stream</strong>
              <span>WavLM Embeddings + Conformer Layers</span>
            </div>
            <div className="flow-connector">➡️</div>
            <div className="flow-node">
              <strong>📝 Linguistic Stream</strong>
              <span>ClinicalBERT + Gated Transformer</span>
            </div>
            <div className="flow-connector">➡️</div>
            <div className="flow-node">
              <strong>🔀 Fusion Layer</strong>
              <span>Cross-Modal Attention (Q/K/V alignment)</span>
            </div>
            <div className="flow-connector">➡️</div>
            <div className="flow-node">
              <strong>🎯 Classifier Head</strong>
              <span>FC Layers + Sigmoid Decision Block</span>
            </div>
          </div>
        </section>

      </main>
    </div>
  );
}

export default App;
