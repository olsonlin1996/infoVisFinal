let selectedCircuitId = null;
let activeCircuitIds = new Set();
let raceDataMap = new Map(); 

const calendar2024 = [
    "bahrain", "jeddah", "albert_park", "suzuka", "shanghai", "miami",
    "imola", "monaco", "villeneuve", "catalunya", "red_bull_ring", "silverstone",
    "hungaroring", "spa", "zandvoort", "monza", "baku", "marina_bay",
    "americas", "rodriguez", "interlagos", "las_vegas", "losail", "yas_marina"
];

const COLOR_ACTIVE = "#e10600";
const COLOR_INACTIVE = "#666666";

const teamColors = {
    "Red Bull Racing": "#3671C6",
    "Red Bull": "#3671C6",
    "Ferrari": "#E80020",
    "Mercedes": "#27F4D2",
    "McLaren": "#FF8000",
    "Aston Martin": "#225941",
    "Alpine": "#0093CC",
    "Williams": "#64C4FF",
    "RB": "#6692FF",
    "Kick Sauber": "#52E252",
    "Haas F1 Team": "#B6BABD",
    "Haas": "#B6BABD"
};

const teamHomeTracksList = [
    { team: "Red Bull", circuitId: "silverstone", label: "Milton Keynes, UK" },
    { team: "Mercedes", circuitId: "silverstone", label: "Brackley, UK" },
    { team: "Ferrari", circuitId: "imola", label: "Maranello, Italy" },
    { team: "McLaren", circuitId: "silverstone", label: "Woking, UK" },
    { team: "Aston Martin", circuitId: "silverstone", label: "Silverstone, UK" },
    { team: "Alpine", circuitId: "silverstone", label: "Enstone, UK / Viry, FR" },
    { team: "Williams", circuitId: "silverstone", label: "Grove, UK" },
    { team: "RB", circuitId: "imola", label: "Faenza, Italy" },
    { team: "Kick Sauber", circuitId: "monza", label: "Hinwil, Switzerland" },
    { team: "Haas", circuitId: "americas", label: "Kannapolis, USA" }
];

function isSameCircuit(id1, id2) {
    if (!id1 || !id2) return false;
    if (id1 === id2) return true;
    if ((id1 === "austin" && id2 === "americas") || (id1 === "americas" && id2 === "austin")) return true;
    return false;
}

const mapContainer = document.getElementById("map-container");
const width = mapContainer.clientWidth;
const height = mapContainer.clientHeight;

const svg = d3.select("#map-container").append("svg")
    .attr("viewBox", `0 0 ${width} ${height}`)
    .style("width", "100%")
    .style("height", "100%")
    .on("wheel", (event) => event.preventDefault());

const tooltip = d3.select("body").append("div")
    .attr("class", "tooltip")
    .style("position", "absolute")
    .style("opacity", 0)
    .style("background", "rgba(0, 0, 0, 0.8)")
    .style("color", "#fff")
    .style("padding", "5px 10px")
    .style("border-radius", "4px")
    .style("font-size", "12px")
    .style("pointer-events", "none")
    .style("border", "1px solid #e10600")
    .style("z-index", "1000");

const defs = svg.append("defs");
const filterShadow = defs.append("filter").attr("id", "track-shadow").attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
filterShadow.append("feGaussianBlur").attr("in", "SourceAlpha").attr("stdDeviation", 4).attr("result", "blur");
filterShadow.append("feOffset").attr("in", "blur").attr("dx", 3).attr("dy", 5).attr("result", "offsetBlur");
const feMerge = filterShadow.append("feMerge");
feMerge.append("feMergeNode").attr("in", "offsetBlur");
feMerge.append("feMergeNode").attr("in", "SourceGraphic");

const glowGradient = defs.append("radialGradient").attr("id", "atmosphereGradient").attr("cx", "50%").attr("cy", "50%").attr("r", "50%");
glowGradient.append("stop").attr("offset", "80%").attr("stop-color", "#4fa3c7").attr("stop-opacity", 0);
glowGradient.append("stop").attr("offset", "100%").attr("stop-color", "#4fa3c7").attr("stop-opacity", 0.3);

const oceanGradient = defs.append("radialGradient").attr("id", "oceanGradient").attr("cx", "50%").attr("cy", "50%").attr("r", "50%");
oceanGradient.append("stop").attr("offset", "0%").attr("stop-color", "#1a2a3a");
oceanGradient.append("stop").attr("offset", "100%").attr("stop-color", "#08101a");

const initialScale = Math.min(width, height) / 2.3;
const maxZoomScale = 6000;
const minZoomScale = 200;
const clickZoomScale = 2500;
const trackThreshold = 500;

const projection = d3.geoOrthographic()
    .scale(initialScale)
    .translate([width / 2, height / 2])
    .clipAngle(90);

const pathGenerator = d3.geoPath().projection(projection);

const minimapWidth = 240;
const minimapHeight = 120;
const minimapSvg = d3.select("#minimap-container").append("svg")
    .attr("width", "100%").attr("height", "100%").attr("viewBox", `0 0 ${minimapWidth} ${minimapHeight}`);
const minimapProjection = d3.geoEquirectangular().scale(38).translate([minimapWidth / 2, minimapHeight / 2]);
const minimapPath = d3.geoPath().projection(minimapProjection);

let ocean, countryPaths, circuitGlyphs, atmosphere, minimapMarker;
let worldJson, f1Json;

Promise.all([
    d3.json("./data/world.json"),
    d3.json("./data/circuit.json"),
    d3.json("./data/f1_2024_weather_and_results.json")
]).then(([world, f1, raceResults]) => {
    worldJson = world;
    f1Json = f1;

    initData(raceResults);
    drawMap();
    drawMinimap();
}).catch(err => console.error("Error loading data:", err));
function createStars() {
    generateBoxShadows(700, "stars");
    generateBoxShadows(200, "stars2");
    generateBoxShadows(100, "stars3");
}

function generateBoxShadows(n, id) {
    const element = document.getElementById(id);
    if (!element) return;

    let boxShadows = [];
    for (let i = 0; i < n; i++) {
        const x = Math.floor(Math.random() * 300 - 150); 
        const y = Math.floor(Math.random() * 300 - 150); 
        
        boxShadows.push(`${x}vw ${y}vh #FFF`);
    }
    element.style.boxShadow = boxShadows.join(", ");
}

createStars();
function initData(raceResults) {
    calendar2024.forEach(id => activeCircuitIds.add(id));

    if (raceResults && Array.isArray(raceResults)) {
        raceResults.forEach(race => {
            const circuitId = calendar2024[race.Round - 1];
            if (circuitId) {
                raceDataMap.set(circuitId, race);
            }
        });
    }
}

function getRaceWeatherSummary(weatherSummary) {
    if (!weatherSummary) return "No Data";

    const avgAir = weatherSummary.AirTemp;
    const avgTrack = weatherSummary.TrackTemp;
    const isRainy = weatherSummary.IsRainy;

    const weatherIcon = isRainy ? "🌧️" : "☀️";
    const status = isRainy ? "Rain" : "Dry";

    return `${weatherIcon} ${status} | Air ${avgAir}°C | Track ${avgTrack}°C`;
}

function drawMap() {
    const countries = topojson.feature(worldJson, worldJson.objects.countries);

    ocean = svg.append("path")
        .datum({ type: "Sphere" })
        .attr("d", pathGenerator)
        .attr("fill", "url(#oceanGradient)")
        .attr("stroke", "#333").attr("stroke-width", 0.5);

    countryPaths = svg.append("g")
        .selectAll("path")
        .data(countries.features)
        .join("path")
        .attr("d", pathGenerator)
        .attr("fill", "#2a3b4d").attr("stroke", "#3e5266").attr("stroke-width", 0.5);

    atmosphere = svg.append("circle")
        .attr("cx", width / 2).attr("cy", height / 2).attr("r", initialScale)
        .style("fill", "url(#atmosphereGradient)").attr("pointer-events", "none");

    let circuits = f1Json.MRData.CircuitTable.Circuits;
    const trackPromises = circuits.map(c => d3.json(`./data/tracks/${c.circuitId}.geojson`).catch(() => null));

    Promise.all(trackPromises).then(trackDataArray => {
        const circuitGeoData = new Map(trackDataArray.map((geoData, i) => [circuits[i].circuitId, geoData]));

        circuitGlyphs = svg.append("g")
            .selectAll("g.circuitGlyph")
            .data(circuits)
            .join("g")
            .attr("class", "circuitGlyph");

        circuitGlyphs.each(function (d) {
            let isActive = activeCircuitIds.has(d.circuitId);
            if (d.circuitId === "austin" && activeCircuitIds.has("americas")) isActive = true;

            const baseColor = isActive ? COLOR_ACTIVE : COLOR_INACTIVE;

            d3.select(this).append("circle")
                .attr("class", "track-dot")
                .attr("r", isActive ? 4.5 : 2.5)
                .attr("fill", baseColor)
                .attr("stroke", isActive ? "rgba(0,0,0,0.5)" : "none")
                .attr("stroke-width", 1)
                .style("opacity", isActive ? 1 : 0.6);

            const geoData = circuitGeoData.get(d.circuitId);
            if (geoData) {
                const glyphSize = 50;
                const glyphProj = d3.geoMercator().fitSize([glyphSize, glyphSize], geoData);
                const glyphPathGen = d3.geoPath().projection(glyphProj);

                d3.select(this).append("path")
                    .attr("class", "track-path")
                    .datum(geoData)
                    .attr("d", glyphPathGen)
                    .attr("fill", "none")
                    .attr("stroke", baseColor)
                    .attr("stroke-width", 3)
                    .attr("stroke-linecap", "round")
                    .attr("stroke-linejoin", "round")
                    .attr("transform", `translate(${-glyphSize / 2}, ${-glyphSize / 2})`)
                    .style("filter", "url(#track-shadow)")
                    .style("display", "none");
            }
        });

        circuitGlyphs
            .on("mouseover", function (event, d) {
                tooltip.style("opacity", 1).html(d.circuitName);
                d3.select(this).style("cursor", "pointer");
                highlightGlyph(d3.select(this));
            })
            .on("mousemove", event => {
                tooltip.style("left", (event.pageX + 15) + "px")
                    .style("top", (event.pageY - 10) + "px");
            })
            .on("mouseout", function (event, d) {
                tooltip.style("opacity", 0);
                refresh();
            })
            .on("click", function (event, d) {
                event.stopPropagation();
                focusOnCircuit(d);
            });

        renderTeamList(circuits);
        refresh();
    });

    const drag = d3.drag()
        .on("drag", (event) => {
            const rotate = projection.rotate();
            const k = 75 / projection.scale();
            projection.rotate([rotate[0] + event.dx * k, rotate[1] - event.dy * k]);
            refresh();
        });
    svg.call(drag);

    const zoom = d3.zoom()
        .scaleExtent([minZoomScale, maxZoomScale])
        .on("zoom", (event) => {
            projection.scale(event.transform.k);
            atmosphere.attr("r", event.transform.k);
            refresh();
        });
    svg.call(zoom);
}

function highlightGlyph(selection) {
    const currentScale = Math.max(1, projection.scale() / 400);
    const hoverStrokeWidth = 12 / currentScale;

    selection.select(".track-dot")
        .attr("r", 8)
        .attr("stroke", "#fff")
        .attr("stroke-width", 2)
        .attr("opacity", 1);

    selection.select(".track-path")
        .attr("stroke-width", hoverStrokeWidth)
        .attr("opacity", 1);
}

function drawMinimap() {
    const countries = topojson.feature(worldJson, worldJson.objects.countries);
    minimapSvg.append("g").selectAll("path").data(countries.features).join("path")
        .attr("d", minimapPath).attr("fill", "#666").attr("stroke", "none");
    minimapMarker = minimapSvg.append("circle")
        .attr("r", 4).attr("fill", "#e10600").attr("stroke", "#fff").attr("stroke-width", 1);
}

function refresh() {
    if (ocean) ocean.attr("d", pathGenerator);
    if (countryPaths) countryPaths.attr("d", pathGenerator);

    const currentK = projection.scale();
    const center = projection.invert([width / 2, height / 2]);

    if (minimapMarker && center) {
        let long = center[0] % 360;
        const xy = minimapProjection([long, center[1]]);
        if (xy) minimapMarker.attr("cx", xy[0]).attr("cy", xy[1]);
    }

    if (circuitGlyphs) {
        circuitGlyphs.attr("transform", d => {
            const coords = projection([+d.Location.long, +d.Location.lat]);
            return coords ? `translate(${coords[0]}, ${coords[1]})` : "translate(-9999,-9999)";
        });

        circuitGlyphs.style("display", d => {
            const dGeo = d3.geoDistance(center, [+d.Location.long, +d.Location.lat]);
            return (dGeo > 1.57) ? "none" : "block";
        });

        if (currentK > trackThreshold) {
            circuitGlyphs.selectAll(".track-dot").style("display", "none");

            let baseScale = Math.max(1, currentK / 400);

            circuitGlyphs.selectAll(".track-path")
                .style("display", "block")
                .attr("stroke", d => {
                    let isActive = activeCircuitIds.has(d.circuitId);
                    if (d.circuitId === "austin" && activeCircuitIds.has("americas")) isActive = true;
                    return isActive ? COLOR_ACTIVE : COLOR_INACTIVE;
                })
                .style("opacity", d => {
                    if (selectedCircuitId) {
                        return isSameCircuit(d.circuitId, selectedCircuitId) ? 1.0 : 0.3;
                    }
                    return 0.8;
                })
                .attr("transform", d => {
                    let finalScale = baseScale;
                    if (selectedCircuitId && !isSameCircuit(d.circuitId, selectedCircuitId)) {
                        finalScale = baseScale * 0.5;
                    }
                    return `scale(${finalScale}) translate(-25, -25)`;
                })
                .attr("stroke-width", d => {
                    let finalScale = baseScale;
                    if (selectedCircuitId && !isSameCircuit(d.circuitId, selectedCircuitId)) {
                        finalScale = baseScale * 0.5;
                    }
                    if (selectedCircuitId && isSameCircuit(d.circuitId, selectedCircuitId)) {
                        return 8 / finalScale;
                    } else {
                        return 2 / finalScale;
                    }
                });

            if (selectedCircuitId) {
                circuitGlyphs.filter(d => isSameCircuit(d.circuitId, selectedCircuitId)).raise();
            }

        } else {
            circuitGlyphs.selectAll(".track-dot").style("display", "block")
                .style("opacity", d => {
                    if (selectedCircuitId) {
                        return isSameCircuit(d.circuitId, selectedCircuitId) ? 1.0 : 0.3;
                    }
                    let isActive = activeCircuitIds.has(d.circuitId);
                    if (d.circuitId === "austin" && activeCircuitIds.has("americas")) isActive = true;
                    return isActive ? 1.0 : 0.6;
                })
                .attr("fill", d => {
                    let isActive = activeCircuitIds.has(d.circuitId);
                    if (d.circuitId === "austin" && activeCircuitIds.has("americas")) isActive = true;
                    return isActive ? COLOR_ACTIVE : COLOR_INACTIVE;
                })
                .attr("stroke", "none")
                .attr("r", d => {
                    let isActive = activeCircuitIds.has(d.circuitId);
                    if (d.circuitId === "austin" && activeCircuitIds.has("americas")) isActive = true;
                    return isActive ? 4.5 : 2.5;
                });
            circuitGlyphs.selectAll(".track-path").style("display", "none");
        }
    }
}

function focusOnCircuit(d) {
    selectedCircuitId = d.circuitId;
    updateInfoPanel(d);

    const targetRotate = [-d.Location.long, -d.Location.lat];
    const targetScale = clickZoomScale;

    d3.transition()
        .duration(1500)
        .tween("rotateZoom", () => {
            const r = d3.interpolate(projection.rotate(), targetRotate);
            const s = d3.interpolate(projection.scale(), targetScale);
            return (t) => {
                projection.rotate(r(t));
                projection.scale(s(t));
                atmosphere.attr("r", s(t));
                refresh();
            };
        });
}

function updateInfoPanel(d) {
    document.getElementById('info-name').innerText = d.circuitName;
    document.getElementById('info-country').innerText = `${d.Location.locality}, ${d.Location.country}`;

    const btn = document.getElementById('enter-track-btn');
    btn.style.display = 'block';
    btn.onclick = (e) => { e.preventDefault(); showTrackView(d); };

    const resultList = document.getElementById('podium-list');
    resultList.innerHTML = '';

    let searchId = d.circuitId;
    if (d.circuitId === "austin") searchId = "americas";
    if (d.circuitId === "americas") searchId = "americas";

    const raceData = raceDataMap.get(searchId);
    const weatherDiv = document.getElementById('weather-info');

    if (raceData) {
        const dateStr = raceData.Date ? raceData.Date.split(" ")[0] : "";

        const weatherSummary = getRaceWeatherSummary(raceData.Weather);

        weatherDiv.innerHTML = `
            <div>📅 ${dateStr}</div>
            <div style="margin-top:4px; font-size: 0.9em; color:#ddd;">${weatherSummary}</div>
        `;

        if (raceData.Podium && raceData.Podium.length > 0) {
            raceData.Podium.forEach(res => {
                const row = document.createElement('div');
                row.className = 'podium-row';

                const posClass = `pos-${res.ClassifiedPosition}`;
                const gridPos = parseInt(res.GridPosition);
                const gridText = (isNaN(gridPos) || gridPos === 0) ? "Pit" : `P${gridPos}`;

                row.style.borderLeft = `3px solid ${teamColors[res.TeamName] || '#555'}`;

                row.innerHTML = `
                    <div class="podium-pos ${posClass}">${res.ClassifiedPosition}</div>
                    <div class="podium-driver">
                        ${res.Abbreviation} <span style="font-size:0.8em; opacity:0.7">(${res.TeamName})</span>
                        <br>
                        <span class="podium-grid" style="color:#aaa; font-size:0.8em;">Start: ${gridText}</span>
                    </div>
                `;
                resultList.appendChild(row);
            });
        }
    } else {
        weatherDiv.innerHTML = "";
        if (activeCircuitIds.has(searchId) || activeCircuitIds.has(d.circuitId)) {
            resultList.innerHTML = '<div style="color:#888;text-align:center; padding-top:10px;">Upcoming / No Data</div>';
        } else {
            resultList.innerHTML = '<div style="color:#555;text-align:center; padding-top:10px;">Not in 2024 Calendar</div>';
        }
    }
}

function renderTeamList(allCircuits) {
    const listDiv = document.getElementById('team-list-content');
    listDiv.innerHTML = "";

    teamHomeTracksList.forEach(item => {
        let searchId = item.circuitId;
        if (item.circuitId === "americas") {
            const tryAustin = allCircuits.find(c => c.circuitId === "austin");
            if (tryAustin) searchId = "austin";
        }

        const circuit = allCircuits.find(c => c.circuitId === searchId);
        if (!circuit) return;

        const div = document.createElement('div');
        div.className = 'team-item';
        div.style.borderLeftColor = teamColors[item.team] || "#fff";

        div.innerHTML = `
            <div class="team-name" style="color:${teamColors[item.team] || '#ddd'}">${item.team}</div>
            <div class="team-track">${item.label}</div>
        `;

        div.onclick = () => focusOnCircuit(circuit);

        div.onmouseenter = () => {
            if (circuitGlyphs) {
                const target = circuitGlyphs.filter(d => isSameCircuit(d.circuitId, circuit.circuitId));
                if (!target.empty()) {
                    highlightGlyph(target);
                }
            }
        };

        div.onmouseleave = () => {
            refresh();
        };

        listDiv.appendChild(div);
    });
}

d3.select('.resetButton').on('click', () => {
    selectedCircuitId = null;
    const targetRotate = [0, 0];
    const targetScale = initialScale;

    document.getElementById('info-name').innerText = " ";
    document.getElementById('info-country').innerText = "Select a track on the globe";
    document.getElementById('enter-track-btn').style.display = 'none';
    document.getElementById('podium-list').innerHTML = '<div style="color:#666;text-align:center;">Select a red point to see results</div>';
    document.getElementById('weather-info').innerHTML = "";

    d3.transition()
        .duration(1200)
        .tween("reset", () => {
            const r = d3.interpolate(projection.rotate(), targetRotate);
            const s = d3.interpolate(projection.scale(), targetScale);
            return (t) => {
                projection.rotate(r(t));
                projection.scale(s(t));
                atmosphere.attr("r", s(t));
                refresh();
            };
        });
});

function showTrackView(d) {
    document.querySelector('.sectionWorldMap').style.display = 'none';
    document.querySelector('.trackContainer').style.display = 'flex';
    d3.select(".track").html(`<h2 style='color:white;text-align:center;margin-top:50px;'>${d.circuitName}</h2>`);
}

document.querySelector('.Totop').addEventListener('click', () => {
    document.querySelector('.trackContainer').style.display = 'none';
    document.querySelector('.sectionWorldMap').style.display = 'flex';
});