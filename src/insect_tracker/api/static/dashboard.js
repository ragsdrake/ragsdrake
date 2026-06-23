const speciesChart = new Chart(document.getElementById("speciesChart"), {
  type: "bar",
  data: { labels: [], datasets: [{ label: "Anzahl Individuen (Tracks)", data: [], backgroundColor: "#2f6b3a" }] },
  options: { responsive: true, scales: { y: { beginAtZero: true } } },
});

async function refreshSpeciesCounts() {
  const res = await fetch("/api/species-counts");
  const rows = await res.json();
  speciesChart.data.labels = rows.map((r) => r.species);
  speciesChart.data.datasets[0].data = rows.map((r) => r.track_count);
  speciesChart.update();
}

async function refreshSightings() {
  const res = await fetch("/api/sightings?limit=50");
  const rows = await res.json();
  const tbody = document.querySelector("#sightingsTable tbody");
  tbody.innerHTML = rows
    .map((r) => {
      const time = new Date(r.timestamp * 1000).toLocaleTimeString();
      return `<tr><td>${time}</td><td>${r.species}</td><td>${r.track_id}</td><td>${r.confidence.toFixed(2)}</td><td>${r.source}</td></tr>`;
    })
    .join("");
}

function refreshAll() {
  refreshSpeciesCounts();
  refreshSightings();
}

document.getElementById("uploadForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  const fileInput = document.getElementById("fileInput");
  const status = document.getElementById("uploadStatus");
  if (!fileInput.files.length) return;

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  status.textContent = "Verarbeite Video ...";
  const res = await fetch("/api/upload", { method: "POST", body: formData });
  const data = await res.json();
  status.textContent = `Fertig: ${data.filename} (max. ${data.max_concurrent_tracks} gleichzeitige Tracks)`;
  refreshAll();
});

refreshAll();
setInterval(refreshAll, 5000);
