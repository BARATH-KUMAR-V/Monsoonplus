/**
 * Watches conditions and raises an alert when they turn bad.
 *
 * Alert intelligence, such as it is: an alert fires on a RISE in severity, not on a
 * state. Something that has been severe for an hour does not re-notify, because an
 * app that buzzes every five minutes during a storm is one the user switches off --
 * and then misses the alert that mattered.
 *
 * Runs only while a tab is open. Waking a closed browser needs a push server, which
 * this build does not have, and the Settings page says so rather than implying it.
 */
import { useEffect, useRef } from 'react';
import { useApp } from '@/data/consumer';

const RANK = { info: 1, warning: 2, severe: 3 };

export default function AlertWatcher() {
  const { alertsEnabled, voiceEnabled, severe, anomalies, focus } = useApp();
  const seen = useRef({});

  useEffect(() => {
    if (!alertsEnabled) return;

    const items = [
      ...severe.map((s) => ({ key: `${s.kind}:${focus?.id || focus?.name}`, rank: RANK[s.severity] || 1, ...s })),
      ...anomalies.filter((a) => a.severity !== 'info').map((a) => ({ key: `anom:${a.kind}`, rank: RANK[a.severity] || 1, ...a })),
    ];

    items.forEach((item) => {
      const before = seen.current[item.key] ?? 0;
      if (item.rank <= before) return;
      seen.current[item.key] = item.rank;

      if ('Notification' in window && Notification.permission === 'granted') {
        try {
          new Notification(item.title, { body: item.detail, tag: item.key });
        } catch {
          /* the OS can refuse; the in-app banner still carries the message */
        }
      }

      if (voiceEnabled && 'speechSynthesis' in window && item.rank >= RANK.severe) {
        try {
          const utterance = new SpeechSynthesisUtterance(`${item.title}. ${item.detail}`);
          utterance.rate = 0.95;
          window.speechSynthesis.speak(utterance);
        } catch {
          /* speech is a nicety */
        }
      }
    });

    // Let a condition that has cleared notify again if it comes back.
    Object.keys(seen.current).forEach((key) => {
      if (!items.some((i) => i.key === key)) delete seen.current[key];
    });
  }, [alertsEnabled, voiceEnabled, severe, anomalies, focus]);

  return null;
}
