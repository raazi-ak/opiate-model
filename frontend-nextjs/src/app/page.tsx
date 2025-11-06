'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, FileText, Brain, Loader2 } from 'lucide-react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Navbar } from '@/components/navbar';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

const API_BASE = 'http://localhost:8000';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  references?: string[];
  timestamp: Date;
  think?: string;
}

interface UploadProgress {
  isUploading: boolean;
  isIngesting: boolean;
  progress: number;
  status: string;
}

export default function ChatBot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState<UploadProgress>({
    isUploading: false,
    isIngesting: false,
    progress: 0,
    status: ''
  });
  const [objective, setObjective] = useState('');
  const [userId, setUserId] = useState('raazifaisal710729');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [stressData, setStressData] = useState<{
    level: 'low' | 'medium' | 'high' | null;
    numeric: number | null;
    heartRate: number | null;
    timestamp: number | null;
    isLive?: boolean;
    isCached?: boolean;
    cacheAge?: number | null;
  }>({ level: null, numeric: null, heartRate: null, timestamp: null, isLive: false, isCached: false, cacheAge: null });
  const [files, setFiles] = useState<File[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [dark, setDark] = useState(false);
  
  useEffect(() => {
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    console.log('New session started:', newSessionId);
  }, []);

  useEffect(() => {
    const saved = localStorage.getItem('theme-dark');
    const prefers = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const d = saved ? saved === '1' : prefers;
    setDark(d);
    document.documentElement.classList.toggle('dark', d);
  }, []);

  const toggleDark = () => {
    setDark((d) => {
      const nd = !d;
      localStorage.setItem('theme-dark', nd ? '1' : '0');
      document.documentElement.classList.toggle('dark', nd);
      return nd;
    });
  };

  const splitThink = (text: string): { visible: string; think: string } => {
    if (!text) return { visible: '', think: '' };
    let visible = '';
    let think = '';
    let i = 0;
    while (i < text.length) {
      const start = text.indexOf('<think>', i);
      if (start === -1) {
        visible += text.slice(i);
        break;
      }
      visible += text.slice(i, start);
      const end = text.indexOf('</think>', start + 7);
      if (end === -1) {
        think += text.slice(start + 7);
        break;
      } else {
        think += text.slice(start + 7, end);
        i = end + 8;
      }
    }
    return { visible, think };
  };

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const messagesContainerRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    // Use setTimeout to ensure DOM is updated
    setTimeout(() => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
      } else if (messagesContainerRef.current) {
        messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
      }
    }, 100);
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading]);

  useEffect(() => {
    const fetchStressData = async () => {
      if (!userId) return;
      
      try {
        const response = await fetch(`${API_BASE}/stress/${userId}?t=${Date.now()}`);
        const data = await response.json();
        
        if (data) {
          setStressData(prev => {
            const newHeartRate = data.heart_rate || null;
            const newStressLevel = data.stress_level as 'low' | 'medium' | 'high' | null;
            
            if (prev.heartRate !== newHeartRate || prev.level !== newStressLevel) {
              return {
                level: newStressLevel,
                numeric: data.stress_level_numeric || null,
                heartRate: newHeartRate,
                timestamp: data.timestamp || null,
                isLive: data.is_live || false,
                isCached: data.is_cached || false,
                cacheAge: data.cache_age_seconds || null
              };
            }
            
            return {
              ...prev,
              numeric: data.stress_level_numeric || null,
              timestamp: data.timestamp || null,
              isLive: data.is_live || false,
              isCached: data.is_cached || false,
              cacheAge: data.cache_age_seconds || null
            };
          });
        }
      } catch (error) {
        console.error('Error fetching stress data:', error);
      }
    };

    fetchStressData();
    const interval = setInterval(fetchStressData, 2000);
    return () => clearInterval(interval);
  }, [userId]);

  const handleFileUpload = async (selectedFiles: FileList | null) => {
    if (!selectedFiles || selectedFiles.length === 0) return;

    const fileArray = Array.from(selectedFiles);
    setFiles(fileArray);
    setUploadProgress({
      isUploading: true,
      isIngesting: false,
      progress: 0,
      status: 'Uploading files...'
    });

    try {
      const formData = new FormData();
      fileArray.forEach(file => {
        formData.append('files', file);
      });

      setUploadProgress(prev => ({ ...prev, progress: 30, status: 'Uploading to server...' }));
      const uploadResponse = await axios.post(`${API_BASE}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      
      // Get the list of saved file names from the upload response
      const savedFiles = uploadResponse.data?.saved || fileArray.map(f => f.name);

      setUploadProgress(prev => ({ 
        ...prev, 
        progress: 60, 
        status: 'Processing documents...',
        isUploading: false,
        isIngesting: true
      }));

      const ingestFormData = new FormData();
      if (sessionId) {
        ingestFormData.append('session_id', sessionId);
      }
      ingestFormData.append('clear_existing', 'false');
      // Pass only the uploaded file names to ingest
      ingestFormData.append('file_names', savedFiles.join(','));
      
      const ingestResponse = await axios.post(`${API_BASE}/ingest`, ingestFormData);
      const returnedSessionId = ingestResponse.data?.session_id;
      if (returnedSessionId && !sessionId) {
        setSessionId(returnedSessionId);
      }
      
      setUploadProgress(prev => ({ 
        ...prev, 
        progress: 100, 
        status: 'Ready to chat!',
        isIngesting: false
      }));

      addMessage('assistant', `Successfully uploaded and processed ${fileArray.length} file(s)! You can now ask questions about your documents.`);

      setTimeout(() => {
        setUploadProgress({
          isUploading: false,
          isIngesting: false,
          progress: 0,
          status: ''
        });
      }, 2000);

    } catch (error) {
      console.error('Upload error:', error);
      setUploadProgress({
        isUploading: false,
        isIngesting: false,
        progress: 0,
        status: 'Upload failed'
      });
      addMessage('assistant', 'Sorry, there was an error uploading your files. Please try again.');
    }
  };

  const addMessage = (role: 'user' | 'assistant', content: string, references?: string[]) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      role,
      content,
      references,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newMessage]);
  };

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');
    addMessage('user', userMessage);
    setIsLoading(true);

    try {
      const formData = new FormData();
      formData.append('message', userMessage);
      if (objective) formData.append('objective', objective);
      formData.append('k', '5');
      if (userId) formData.append('user_id', userId);
      if (sessionId) formData.append('session_id', sessionId);

      const response = await fetch(`${API_BASE}/chat/stream`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok || !response.body) {
        throw new Error('Streaming failed');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();

      const msgId = Date.now().toString();
      setMessages(prev => [...prev, { id: msgId, role: 'assistant', content: '', timestamp: new Date(), think: '' }]);

      let done = false;
      let inThink = false;
      let buffer = '';
      while (!done) {
        const { value, done: doneReading } = await reader.read();
        done = doneReading;
        const chunkValue = decoder.decode(value || new Uint8Array(), { stream: true });
        if (!chunkValue) continue;
        buffer += chunkValue;

        let i = 0;
        let visibleAppend = '';
        let thinkAppend = '';
        while (i < buffer.length) {
          if (!inThink) {
            const start = buffer.indexOf('<think>', i);
            if (start === -1) {
              visibleAppend += buffer.slice(i);
              buffer = '';
              break;
            } else {
              visibleAppend += buffer.slice(i, start);
              i = start + '<think>'.length;
              inThink = true;
            }
          } else {
            const end = buffer.indexOf('</think>', i);
            if (end === -1) {
              thinkAppend += buffer.slice(i);
              buffer = '';
              break;
            } else {
              thinkAppend += buffer.slice(i, end);
              i = end + '</think>'.length;
              inThink = false;
              if (i >= buffer.length) {
                buffer = '';
                break;
              }
            }
          }
        }

        if (visibleAppend || thinkAppend) {
          setMessages(prev => prev.map(m => m.id === msgId
            ? { ...m, content: m.content + visibleAppend, think: (m.think || '') + thinkAppend }
            : m
          ));
        }
      }

      const refForm = new FormData();
      refForm.append('message', userMessage);
      if (objective) refForm.append('objective', objective);
      refForm.append('k', '5');
      if (userId) refForm.append('user_id', userId);
      if (sessionId) refForm.append('session_id', sessionId);
      const refsResp = await axios.post(`${API_BASE}/chat`, refForm);
      const { references, answer } = refsResp.data || {};
      const parts = splitThink(answer || '');
      setMessages(prev => prev.map(m => m.id === msgId ? { ...m, content: parts.visible || m.content, think: (m.think || '') || parts.think, references: references?.map((r:any)=> r.source || r) } : m));

    } catch (error) {
      console.error('Chat error:', error);
      addMessage('assistant', 'Sorry, I encountered an error. Please make sure the backend server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleNewSession = async () => {
    if (!confirm('Start a new session? This will clear all current messages and documents.')) {
      return;
    }

    const oldSessionId = sessionId;
    
    // Generate new session ID
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    setSessionId(newSessionId);
    
    // Clear messages
    setMessages([]);
    setObjective('');
    
    // Clear backend indexes for old session if it exists
    if (oldSessionId) {
      try {
        const formData = new FormData();
        formData.append('session_id', oldSessionId);
        await axios.post(`${API_BASE}/session/clear`, formData);
        console.log('Cleared old session:', oldSessionId);
      } catch (error) {
        console.error('Error clearing old session:', error);
      }
    }
    
    console.log('New session started:', newSessionId);
  };

  return (
    <div className="flex flex-col h-screen bg-background overflow-hidden">
      <Navbar
        userId={userId}
        onUserIdChange={setUserId}
        stressData={{ level: stressData.level, heartRate: stressData.heartRate }}
        sessionId={sessionId}
        onFileUpload={() => fileInputRef.current?.click()}
        onNewSession={handleNewSession}
        dark={dark}
        onToggleDark={toggleDark}
      />

      <input
        ref={fileInputRef}
        type="file"
        multiple
        accept=".pdf,.txt,.md"
        onChange={(e) => handleFileUpload(e.target.files)}
        className="hidden"
      />

      {/* Upload Progress */}
      {uploadProgress.isUploading || uploadProgress.isIngesting ? (
        <div className="border-b bg-card">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center gap-3">
              <Loader2 className="h-5 w-5 animate-spin text-primary" />
              <div className="flex-1">
                <div className="flex justify-between text-sm mb-1">
                  <span>{uploadProgress.status}</span>
                  <span>{uploadProgress.progress}%</span>
                </div>
                <div className="w-full bg-secondary rounded-full h-2">
                  <div
                    className="bg-primary h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress.progress}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      {/* Objective Badge */}
      {objective && (
        <div className="border-b bg-muted/50">
          <div className="container mx-auto px-4 py-3">
            <div className="flex items-center gap-2">
              <FileText className="h-4 w-4 text-primary" />
              <span className="text-sm">
                <strong>Objective:</strong> {objective}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Chat Messages */}
      <div className="flex-1 overflow-hidden min-h-0">
        <div 
          ref={messagesContainerRef}
          className="container mx-auto px-4 py-6 max-w-4xl h-full overflow-y-auto scroll-smooth"
        >
          <div className="space-y-6 pb-24">
            {messages.length === 0 ? (
            <Card className="border-dashed">
              <CardContent className="flex flex-col items-center justify-center py-12">
                <div className="flex h-16 w-16 items-center justify-center rounded-full bg-primary/10 mb-4">
                  <Brain className="h-8 w-8 text-primary" />
                </div>
                <h3 className="text-lg font-semibold mb-2">Welcome to Study Assistant</h3>
                <p className="text-muted-foreground mb-6 text-center">Upload your study materials and start chatting.</p>
                
                <div className="space-y-3">
                  <Button
                    onClick={() => {
                      const obj = prompt('What do you want to learn or focus on?');
                      if (obj) setObjective(obj);
                    }}
                  >
                    Set Learning Objective
                  </Button>
                  <p className="text-sm text-muted-foreground text-center">or upload documents to get started</p>
                </div>
              </CardContent>
            </Card>
          ) : (
            messages.map((message) => (
              <div
                key={message.id}
                className={cn("flex", message.role === 'user' ? 'justify-end' : 'justify-start')}
              >
                <Card className={cn(
                  "max-w-3xl",
                  message.role === 'user' ? 'bg-primary text-primary-foreground' : ''
                )}>
                  <CardContent className="p-4">
                    <div className="prose prose-sm dark:prose-invert max-w-none">
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                          // Custom styling for markdown elements
                          h1: ({...props}: any) => <h1 className="text-2xl font-bold mt-6 mb-4" {...props} />,
                          h2: ({...props}: any) => <h2 className="text-xl font-semibold mt-5 mb-3" {...props} />,
                          h3: ({...props}: any) => <h3 className="text-lg font-semibold mt-4 mb-2" {...props} />,
                          h4: ({...props}: any) => <h4 className="text-base font-semibold mt-3 mb-2" {...props} />,
                          p: ({...props}: any) => <p className="mb-4 leading-relaxed" {...props} />,
                          ul: ({...props}: any) => <ul className="list-disc list-inside mb-4 space-y-1 ml-4" {...props} />,
                          ol: ({...props}: any) => <ol className="list-decimal list-inside mb-4 space-y-1 ml-4" {...props} />,
                          li: ({...props}: any) => <li className="mb-1" {...props} />,
                          blockquote: ({...props}: any) => <blockquote className="border-l-4 border-primary pl-4 italic my-4 text-muted-foreground" {...props} />,
                          code: ({className, ...props}: any) => {
                            const isInline = !className;
                            return isInline ? (
                              <code className="bg-muted px-1.5 py-0.5 rounded text-sm font-mono" {...props} />
                            ) : (
                              <code className="block bg-muted p-3 rounded-lg text-sm font-mono overflow-x-auto my-4" {...props} />
                            );
                          },
                          pre: ({...props}: any) => <pre className="bg-muted p-3 rounded-lg overflow-x-auto my-4" {...props} />,
                          strong: ({...props}: any) => <strong className="font-semibold" {...props} />,
                          em: ({...props}: any) => <em className="italic" {...props} />,
                          hr: ({...props}: any) => <hr className="my-6 border-border" {...props} />,
                          table: ({...props}: any) => <table className="w-full border-collapse border border-border my-4" {...props} />,
                          thead: ({...props}: any) => <thead className="bg-muted" {...props} />,
                          tbody: ({...props}: any) => <tbody {...props} />,
                          tr: ({...props}: any) => <tr className="border-b border-border" {...props} />,
                          th: ({...props}: any) => <th className="border border-border px-4 py-2 text-left font-semibold" {...props} />,
                          td: ({...props}: any) => <td className="border border-border px-4 py-2" {...props} />,
                          a: ({...props}: any) => <a className="text-primary hover:underline" {...props} />,
                        }}
                      >
                        {message.content}
                      </ReactMarkdown>
                    </div>
                    
                    {message.role === 'assistant' && message.think && message.think.trim() && (
                      <details className="mt-2">
                        <summary className="text-xs text-muted-foreground cursor-pointer">Show thinking</summary>
                        <pre className="mt-2 text-xs text-muted-foreground whitespace-pre-wrap">{message.think}</pre>
                      </details>
                    )}
                    
                    {message.references && message.references.length > 0 && (
                      <div className="mt-3 pt-3 border-t">
                        <div className="text-xs text-muted-foreground mb-2">Sources:</div>
                        <div className="flex flex-wrap gap-1">
                          {message.references.map((ref, index) => (
                            <Badge key={index} variant="secondary" className="text-xs">
                              {ref}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            ))
          )}

          {isLoading && (
            <div className="flex justify-start">
              <Card>
                <CardContent className="p-4">
                  <div className="flex items-center gap-2">
                    <Loader2 className="h-4 w-4 animate-spin text-primary" />
                    <span>Thinking...</span>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </div>

      {/* Input Area */}
      <div className="fixed bottom-0 left-0 right-0 border-t bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto px-4 py-4 max-w-4xl">
          <div className="flex items-end gap-3">
            <div className="flex-1">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ask me anything about your study materials..."
                className="flex min-h-[48px] w-full max-h-[120px] rounded-md border border-input bg-background px-3 py-2 text-base ring-offset-background placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:cursor-not-allowed disabled:opacity-50 md:text-sm resize-none"
                rows={1}
                style={{ minHeight: '48px', maxHeight: '120px' }}
              />
            </div>
            <Button
              onClick={handleSendMessage}
              disabled={!input.trim() || isLoading}
              size="icon"
            >
              <Send className="h-5 w-5" />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
