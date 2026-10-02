import { Link } from 'react-router-dom';
import { useStore } from '@/data/store';
import { WATERLOGGING_NOTE } from '@/config/waterlogging';

/**
 * What the app is, what it is not, and where every number comes from. Written for a
 * traveller first and a reviewer second.
 */
export default function About() {
  const { setViewMode } = useStore();

  return (
    <div className="stack cm-page">
      <h1 className="cm-h1">About MonsoonPlus</h1>

      <section className="card pad">
        <h2 className="card-title">What it does</h2>
        <p>
          MonsoonPlus puts weather and travel together. It shows what the weather is doing now and over the next two
          weeks, warns about heavy rain, storms and strong wind, and when you plan a trip it compares the routes on
          travel time, traffic, rain and the chance of water on the road, then explains which one it would take and why.
        </p>
      </section>

      <section className="card pad">
        <h2 className="card-title">Where the numbers come from</h2>
        <dl className="sources">
          <dt>Official forecast</dt>
          <dd>Weather, rain, wind, UV and air quality come from Open-Meteo, a free public weather service. Temperature and rain figures on the Weather page are theirs, not ours.</dd>
          <dt>Roads and routes</dt>
          <dd>Map data from OpenStreetMap. Routes are planned by OSRM, which knows the road network but nothing about weather. The public OSRM service hosts the car road network, so walking and cycling trips follow driving roads.</dd>
          <dt>MonsoonPlus estimate</dt>
          <dd>Flooding risk, expected traffic and travel time are our own transparent rules over real inputs: rainfall, ground height from the Copernicus elevation dataset, time of day, and areas with a history of flooding. These are estimates and are labelled as such. {WATERLOGGING_NOTE}</dd>
          <dt>MonsoonPlus AI prediction</dt>
          <dd>A trained neural network forecasts road speeds 15, 30 and 60 minutes ahead. It was trained on ten roads in the Velachery to OMR area, so it is used only there and labelled separately. It does not cover the rest of the city.</dd>
          <dt>Community report</dt>
          <dd>Hazards reported by users. Unverified, and in this build kept only in your own browser.</dd>
        </dl>
      </section>

      <section className="card pad">
        <h2 className="card-title">What it is not</h2>
        <p>
          MonsoonPlus is a student project and a planning aid. It forecasts conditions; it cannot see the road in front
          of you. It does not issue official weather warnings, and it is not connected to the India Meteorological
          Department or any emergency service. Never use it to decide whether to drive, ride or walk through flood
          water. If you reach standing water, turn back.
        </p>
      </section>

      <section className="card pad">
        <h2 className="card-title">Your privacy</h2>
        <p>
          There is no MonsoonPlus account and no MonsoonPlus server. Saved places, reports and settings live in this
          browser. Your location, if you share it, is used in the browser to find nearby places and is not stored. The
          map, search, routing and weather services receive the coordinates you look up, as they must to answer.
        </p>
      </section>

      <section className="card pad">
        <h2 className="card-title">For reviewers</h2>
        <p className="cm-muted">
          The developer view has the model architecture, training data, evaluation metrics, predicted-versus-actual
          charts and the data-mode switch.
        </p>
        <div className="inline">
          <button type="button" className="btn" onClick={() => setViewMode('developer')}>Open developer view</button>
          <Link className="btn ghost" to="/settings">Settings</Link>
        </div>
      </section>
    </div>
  );
}
