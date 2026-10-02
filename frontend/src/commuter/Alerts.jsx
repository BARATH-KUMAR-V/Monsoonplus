/**
 * What the app is currently warning about, in one place: severe weather, unusual
 * conditions, and the hazards people have reported.
 */
import { Link } from 'react-router-dom';
import { useApp, REPORT_TTL_HOURS } from '@/data/consumer';
import StatusBanner from './StatusBanner';
import SourceBadge from './SourceBadge';
import { relativeTime } from '@/lib/format';

export default function Alerts() {
  const { severe, anomalies, reports, removeReport, alertsEnabled, setAlertsEnabled, focus, places } = useApp();
  const nothing = !severe.length && !anomalies.length && !reports.length;

  const enable = async () => {
    if ('Notification' in window && Notification.permission === 'default') await Notification.requestPermission();
    setAlertsEnabled(true);
  };

  return (
    <div className="stack cm-page">
      <h1 className="cm-h1">Alerts</h1>

      {!alertsEnabled && (
        <section className="card pad">
          <h2 className="card-title">Get told before you leave</h2>
          <p className="cm-muted">
            Turn on alerts and MonsoonPlus will notify you when heavy rain, storms or flooding risk build up around{' '}
            {focus.name}{places.length ? ' or your saved places' : ''}.
          </p>
          <button type="button" className="btn cm-primary" style={{ marginTop: 10 }} onClick={enable}>Turn on alerts</button>
        </section>
      )}

      {nothing && (
        <StatusBanner tier="ok" title="Nothing to warn you about right now." detail={`No severe weather, unusual conditions or reported hazards around ${focus.name}.`} />
      )}

      {severe.map((condition) => (
        <StatusBanner
          key={condition.title}
          tier={condition.severity === 'severe' ? 'critical' : 'warning'}
          title={condition.title}
          detail={condition.detail}
        >
          <SourceBadge source="official" basis="threshold over the Open-Meteo forecast" />
        </StatusBanner>
      ))}

      {anomalies.length > 0 && (
        <section className="card pad">
          <h2 className="card-title">Unusual for this area</h2>
          <ul className="cm-list">
            {anomalies.map((anomaly) => (
              <li key={anomaly.title}>
                <span className={`cm-dot ${anomaly.severity === 'warning' ? 'jammed' : 'slow'}`} aria-hidden="true" />
                <div>
                  <div className="cm-strong">{anomaly.title}</div>
                  <div className="cm-muted">{anomaly.detail}</div>
                </div>
                <SourceBadge source="estimate" basis="compared against the past four weeks here" />
              </li>
            ))}
          </ul>
          <p className="cm-muted small">
            This says what is unusual, not why. A sudden change can have many causes.
          </p>
        </section>
      )}

      <section className="card pad">
        <div className="card-head">
          <h2 className="card-title">Reported hazards</h2>
          <Link className="btn ghost small" to="/report">Report one</Link>
        </div>
        {reports.length === 0 ? (
          <p className="cm-muted">Nothing reported in the last {REPORT_TTL_HOURS} hours.</p>
        ) : (
          <ul className="cm-list">
            {reports.map((report) => (
              <li key={report.id}>
                <span className={`cm-dot ${report.severity === 'severe' ? 'jammed' : 'slow'}`} aria-hidden="true" />
                <div>
                  <div className="cm-strong">{report.label}</div>
                  <div className="cm-muted">
                    {report.placeName || 'Reported location'} · {relativeTime(report.at)}
                    {report.note ? ` · ${report.note}` : ''}
                  </div>
                </div>
                <button type="button" className="btn ghost small" onClick={() => removeReport(report.id)} aria-label={`Remove the ${report.label} report`}>
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
