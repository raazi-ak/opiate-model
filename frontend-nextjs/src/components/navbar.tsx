'use client';

import { useState, useEffect } from 'react';
import { Brain, Moon, SunMedium, Activity, Upload, Settings, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';

interface NavbarProps {
  userId: string;
  onUserIdChange: (userId: string) => void;
  stressData: {
    level: 'low' | 'medium' | 'high' | null;
    heartRate: number | null;
  };
  sessionId: string | null;
  onFileUpload: () => void;
  onNewSession: () => void;
  dark: boolean;
  onToggleDark: () => void;
}

export function Navbar({
  userId,
  onUserIdChange,
  stressData,
  sessionId,
  onFileUpload,
  onNewSession,
  dark,
  onToggleDark,
}: NavbarProps) {
  const getStressLabel = (level: 'low' | 'medium' | 'high' | null) => {
    if (!level) return 'Unknown';
    const labels = {
      low: '😌 Low Stress',
      medium: '😐 Medium Stress',
      high: '😰 High Stress',
    };
    return labels[level];
  };

  const getStressColor = (level: 'low' | 'medium' | 'high' | null) => {
    if (!level) return 'bg-neutral-500';
    const colors = {
      low: 'bg-emerald-500',
      medium: 'bg-yellow-500',
      high: 'bg-red-500',
    };
    return colors[level];
  };

  return (
    <nav className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="container flex h-16 items-center justify-between px-4">
        {/* Left: Logo and Title */}
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-md bg-emerald-500">
            <Brain className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-semibold">Study Assistant</h1>
            <p className="text-xs text-muted-foreground">AI-powered learning companion</p>
          </div>
        </div>

        {/* Center: Stress Level & Session */}
        <div className="flex items-center gap-4">
          {stressData.level && (
            <div className="flex items-center gap-2 rounded-lg border bg-card px-3 py-2">
              <div className={`h-2 w-2 rounded-full ${getStressColor(stressData.level)}`} />
              <div className="text-sm">
                <div className="font-medium">{getStressLabel(stressData.level)}</div>
                {stressData.heartRate && (
                  <div className="text-xs text-muted-foreground">
                    HR: {stressData.heartRate} bpm
                  </div>
                )}
              </div>
            </div>
          )}
          
          {sessionId && (
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="font-mono text-xs" title={sessionId}>
                <span className="hidden md:inline">Session: </span>
                <span className="break-all">{sessionId}</span>
              </Badge>
            </div>
          )}
        </div>

        {/* Right: Actions */}
        <div className="flex items-center gap-2">
          <Input
            type="text"
            value={userId}
            onChange={(e) => onUserIdChange(e.target.value)}
            placeholder="User ID"
            className="w-40 h-9 text-sm"
          />
          
          <Button
            variant="outline"
            size="icon"
            onClick={onToggleDark}
            className="h-9 w-9"
          >
            {dark ? (
              <SunMedium className="h-4 w-4" />
            ) : (
              <Moon className="h-4 w-4" />
            )}
          </Button>
          
          <Button
            onClick={onNewSession}
            variant="outline"
            className="h-9 gap-2"
            title="Start a new session (clears current documents)"
          >
            <RefreshCw className="h-4 w-4" />
            <span className="hidden sm:inline">New Session</span>
          </Button>
          
          <Button
            onClick={onFileUpload}
            className="h-9 gap-2"
            variant="default"
          >
            <Upload className="h-4 w-4" />
            <span className="hidden sm:inline">Upload</span>
          </Button>
        </div>
      </div>
    </nav>
  );
}

