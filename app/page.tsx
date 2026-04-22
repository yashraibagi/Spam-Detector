'use client';

import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";
import { 
  ShieldCheck, 
  AlertTriangle, 
  Mail, 
  Zap, 
  RefreshCcw, 
  Info,
  CheckCircle2,
  XCircle
} from "lucide-react";

interface PredictionResult {
  prediction: 'SPAM' | 'HAM';
  spam_probability: number;
  ham_probability: number;
  input_length: number;
  error?: string;
}

export default function Home() {
  const [emailText, setEmailText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!emailText.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/classify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: emailText }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Failed to analyze email');
      }

      setResult(data);
    } catch (err: any) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setEmailText('');
    setResult(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 py-12 px-4 sm:px-6 lg:px-8 transition-colors duration-300">
      <div className="max-w-3xl mx-auto space-y-8">
        
        {/* Header Section */}
        <div className="text-center space-y-4">
          <div className="inline-flex items-center justify-center p-3 bg-primary/10 rounded-2xl mb-2">
            <ShieldCheck className="w-10 h-10 text-primary" />
          </div>
          <h1 className="text-4xl font-extrabold tracking-tight text-slate-900 dark:text-slate-50 sm:text-5xl">
            Spam Email Detector
          </h1>
          <p className="text-xl text-slate-500 dark:text-slate-400 max-w-2xl mx-auto">
            Advanced machine learning analysis to protect your inbox from unwanted messages.
          </p>
        </div>

        {/* Input Card */}
        <Card className="border-none shadow-xl bg-white dark:bg-slate-900 overflow-hidden ring-1 ring-slate-200 dark:ring-slate-800">
          <CardHeader className="pb-4">
            <CardTitle className="flex items-center gap-2">
              <Mail className="w-5 h-5" />
              Analyze Message
            </CardTitle>
            <CardDescription>
              Paste the content of the email you want to check below.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Textarea 
              placeholder="Paste email content here (e.g., 'Congratulations! You've won a $1000 gift card...')"
              className="min-h-[200px] text-base resize-none focus-visible:ring-primary focus-visible:border-primary border-slate-200 dark:border-slate-800 dark:bg-slate-950/50"
              value={emailText}
              onChange={(e) => setEmailText(e.target.value)}
              disabled={loading}
            />
            
            <div className="flex gap-3 pt-2">
              <Button 
                onClick={handleAnalyze} 
                disabled={loading || !emailText.trim()}
                className="flex-1 h-12 text-lg font-semibold transition-all hover:scale-[1.02] active:scale-[0.98]"
              >
                {loading ? (
                  <>
                    <RefreshCcw className="mr-2 h-5 w-5 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  <>
                    <Zap className="mr-2 h-5 w-5 fill-current" />
                    Classify Email
                  </>
                )}
              </Button>
              {(result || emailText) && !loading && (
                <Button 
                  variant="outline" 
                  onClick={handleReset}
                  className="h-12 px-6"
                >
                  <RefreshCcw className="h-5 w-5" />
                </Button>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Error State */}
        {error && (
          <Alert variant="destructive" className="animate-in fade-in slide-in-from-top-4 duration-300">
            <AlertTriangle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {/* Result Section */}
        {result && (
          <div className="animate-in fade-in slide-in-from-bottom-8 duration-500">
            <Card className={`border-none shadow-2xl overflow-hidden ring-1 ${
              result.prediction === 'SPAM' 
                ? 'ring-red-200 dark:ring-red-900/30' 
                : 'ring-emerald-200 dark:ring-emerald-900/30'
            }`}>
              <div className={`h-2 w-full ${
                result.prediction === 'SPAM' ? 'bg-red-500' : 'bg-emerald-500'
              }`} />
              
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div className="space-y-1">
                    <CardTitle className="text-2xl font-bold flex items-center gap-2">
                      {result.prediction === 'SPAM' ? (
                        <>
                          <AlertTriangle className="text-red-500" />
                          Spam Detected
                        </>
                      ) : (
                        <>
                          <CheckCircle2 className="text-emerald-500" />
                          Legitimate Email
                        </>
                      )}
                    </CardTitle>
                    <CardDescription>
                      Analysis based on TF-IDF vectorization and Logistic Regression.
                    </CardDescription>
                  </div>
                  <Badge 
                    variant={result.prediction === 'SPAM' ? "destructive" : "secondary"}
                    className={`text-sm py-1 px-3 ${
                      result.prediction === 'HAM' ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-400' : ''
                    }`}
                  >
                    {result.prediction}
                  </Badge>
                </div>
              </CardHeader>
              
              <CardContent className="space-y-6">
                <div className="space-y-2">
                  <div className="flex justify-between text-sm font-medium">
                    <span className="text-slate-600 dark:text-slate-400">Model Confidence</span>
                    <span className={result.prediction === 'SPAM' ? 'text-red-600' : 'text-emerald-600'}>
                      {(Math.max(result.spam_probability, result.ham_probability) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <Progress 
                    value={Math.max(result.spam_probability, result.ham_probability) * 100} 
                    className={`h-3 ${
                      result.prediction === 'SPAM' ? '[&>div]:bg-red-500' : '[&>div]:bg-emerald-500'
                    }`}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
                    <p className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wider font-bold mb-1">Spam Score</p>
                    <p className="text-2xl font-mono font-bold text-red-500">{(result.spam_probability * 100).toFixed(2)}%</p>
                  </div>
                  <div className="p-4 rounded-xl bg-slate-50 dark:bg-slate-800/50 border border-slate-100 dark:border-slate-800">
                    <p className="text-xs text-slate-500 dark:text-slate-400 uppercase tracking-wider font-bold mb-1">Ham Score</p>
                    <p className="text-2xl font-mono font-bold text-emerald-500">{(result.ham_probability * 100).toFixed(2)}%</p>
                  </div>
                </div>

                <Separator className="bg-slate-200 dark:bg-slate-800" />
                
                <div className="flex items-start gap-2 text-sm text-slate-500 dark:text-slate-400">
                  <Info className="w-4 h-4 mt-0.5" />
                  <p>
                    Processed {result.input_length} characters. The model predicts this is {result.prediction.toLowerCase()} 
                    with a higher probability of {Math.max(result.spam_probability, result.ham_probability).toFixed(4)}.
                  </p>
                </div>
              </CardContent>
              <CardFooter className="bg-slate-50/50 dark:bg-slate-800/20 py-4 border-t border-slate-100 dark:border-slate-800 flex justify-center">
                <p className="text-xs text-slate-400 flex items-center gap-1 uppercase tracking-tight font-medium">
                  <Zap className="w-3 h-3 h-3" />
                  Real-time ML Inference
                </p>
              </CardFooter>
            </Card>
          </div>
        )}

        {/* Footer Info */}
        {!result && !loading && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4">
            <div className="p-5 rounded-2xl bg-white dark:bg-slate-900 shadow-sm border border-slate-100 dark:border-slate-800 text-center">
              <CheckCircle2 className="w-6 h-6 text-emerald-500 mx-auto mb-3" />
              <h3 className="font-semibold mb-1">High Accuracy</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">Trained on thousands of verified spam/ham samples.</p>
            </div>
            <div className="p-5 rounded-2xl bg-white dark:bg-slate-900 shadow-sm border border-slate-100 dark:border-slate-800 text-center">
              <Zap className="w-6 h-6 text-yellow-500 mx-auto mb-3" />
              <h3 className="font-semibold mb-1">Instant Results</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">Blazing fast inference using lightweight models.</p>
            </div>
            <div className="p-5 rounded-2xl bg-white dark:bg-slate-900 shadow-sm border border-slate-100 dark:border-slate-800 text-center">
              <ShieldCheck className="w-6 h-6 text-blue-500 mx-auto mb-3" />
              <h3 className="font-semibold mb-1">Privacy First</h3>
              <p className="text-sm text-slate-500 dark:text-slate-400">Your email content is processed and never stored.</p>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}