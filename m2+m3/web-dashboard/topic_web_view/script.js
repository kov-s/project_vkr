
let data = [];
const tableBody = document.querySelector("#dataTable tbody");
const topicFilter = document.getElementById("topicFilter");
const searchInput = document.getElementById("searchInput");
const sortSelect = document.getElementById("sortSelect");

fetch("data.csv")
  .then(res => res.text())
  .then(text => {
    const rows = text.trim().split("\n").slice(1);
    data = rows.map(row => {
      const [id, text, processed, hashtags, topic, prob] = row.split(/,(?=(?:[^"]*"[^"]*")*[^"]*$)/);
      return {
        id,
        text: text.replace(/^"|"$/g, ""),
        hashtags: hashtags.replace(/^"|"$/g, ""),
        topic: parseInt(topic),
        prob: parseFloat(prob)
      };
    });

    const uniqueTopics = [...new Set(data.map(d => d.topic))].sort((a, b) => a - b);
    uniqueTopics.forEach(t => {
      const opt = document.createElement("option");
      opt.value = t;
      opt.textContent = t;
      topicFilter.appendChild(opt);
    });

    renderTable(data);
  });

function renderTable(filteredData) {
  tableBody.innerHTML = "";
  filteredData.forEach(d => {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${d.id}</td>
      <td>${d.text}</td>
      <td>${d.topic}</td>
      <td>${d.prob.toFixed(2)}</td>
      <td>${d.hashtags}</td>
    `;
    tableBody.appendChild(row);
  });
}

function applyFilters() {
  let filtered = [...data];

  const search = searchInput.value.toLowerCase();
  const topic = topicFilter.value;

  if (search) {
    filtered = filtered.filter(d => d.text.toLowerCase().includes(search));
  }

  if (topic) {
    filtered = filtered.filter(d => d.topic.toString() === topic);
  }

  if (sortSelect.value === "probability") {
    filtered.sort((a, b) => b.prob - a.prob);
  }

  renderTable(filtered);
}

searchInput.addEventListener("input", applyFilters);
topicFilter.addEventListener("change", applyFilters);
sortSelect.addEventListener("change", applyFilters);
