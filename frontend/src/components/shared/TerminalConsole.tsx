import React, { useRef, useEffect } from "react";
import "./TerminalConsole.css";

interface TerminalConsoleProps {
    logs: string[];
    maxLines?: number;
    height?: string;
}

export function TerminalConsole({
    logs,
    maxLines = 1000,
    height = "300px",
}: TerminalConsoleProps) {
    const containerRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom whenever logs change
    useEffect(() => {
        if (containerRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [logs]);

    // Keep only the tail of the logs
    const displayLogs = logs.slice(-maxLines);

    return (
        <div className="terminal-console-wrapper" style={{ height }}>
            <div className="terminal-header">
                <span className="terminal-title">Runner Output</span>
                <div className="terminal-dots">
                    <span className="dot red"></span>
                    <span className="dot yellow"></span>
                    <span className="dot green"></span>
                </div>
            </div>
            <div className="terminal-body" ref={containerRef}>
                {displayLogs.length === 0 ? (
                    <div className="terminal-empty">Waiting for output...</div>
                ) : (
                    displayLogs.map((line, idx) => {
                        const isError = line.startsWith("[ERR]");
                        const displayLine = isError ? line.replace("[ERR] ", "") : line;
                        return (
                            <div
                                key={idx}
                                className={`terminal-line ${isError ? "error-line" : ""}`}
                            >
                                {/* Optional: we could syntax highlight JSON lines or specific keywords here */}
                                {displayLine}
                            </div>
                        );
                    })
                )}
            </div>
        </div>
    );
}
