import AppKit
import SwiftUI

private let teal = Color(red: 0.06, green: 0.37, blue: 0.40)
private let ink = Color(red: 0.10, green: 0.20, blue: 0.23)
private let mountains = [("alta", "Alta", "Utah"), ("vail", "Vail", "Colorado"),
                         ("mansfield", "Stowe", "Vermont"), ("jay", "Jay Peak", "Vermont")]
private let horizons = [1, 3, 6, 12, 24, 48]
private let baseURL = "http://127.0.0.1:8765"

private func parseDate(_ text: String?) -> Date? {
    guard let text = text else { return nil }
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    if let date = formatter.date(from: text) { return date }
    formatter.formatOptions = [.withInternetDateTime]
    return formatter.date(from: text)
}

private func snowColor(_ inches: Double) -> Color {
    let t = min(max(inches, 0), 1)
    let stops: [[Double]] = [[78,157,166], [66,107,171], [81,53,140]]
    let index = t < 0.5 ? 0 : 1
    let f = t < 0.5 ? t * 2 : (t - 0.5) * 2
    let c = (0..<3).map { (stops[index][$0] * (1-f) + stops[index+1][$0] * f) / 255 }
    return Color(red: c[0], green: c[1], blue: c[2])
}

@MainActor
final class ForecastStore: ObservableObject {
    let autoUpdater = HourlyAutoUpdater()
    @Published var site: String { didSet { UserDefaults.standard.set(site, forKey: "mountain") } }
    @Published var hours: Int { didSet { UserDefaults.standard.set(hours, forKey: "horizon") } }
    @Published var metric: Bool { didSet { UserDefaults.standard.set(metric, forKey: "metric") } }
    @Published var payload: [String: Any]?
    @Published var loading = false
    @Published var refreshing = false
    @Published var message = "Starting local forecast service…"
    @Published var failed = false
    private var serverProcess: Process?
    private var startup: Task<Void, Error>?
    private var requestID = UUID()

    init() {
        let savedSite = UserDefaults.standard.string(forKey: "mountain") ?? "alta"
        site = mountains.contains(where: { $0.0 == savedSite }) ? savedSite : "alta"
        let savedHours = UserDefaults.standard.integer(forKey: "horizon")
        hours = horizons.contains(savedHours) ? savedHours : 48
        metric = UserDefaults.standard.bool(forKey: "metric")
        autoUpdater.start(logURL: projectRoot.appendingPathComponent("outputs/dashboard/widget-auto-update.json"),
                          isBusy: { [weak self] in self?.refreshing != false || self?.loading != false },
                          refresh: { [weak self] in await self?.refresh(automatic: true) })
    }

    var projectRoot: URL {
        let adjacent = Bundle.main.bundleURL.deletingLastPathComponent().deletingLastPathComponent()
        if FileManager.default.fileExists(atPath: adjacent.appendingPathComponent("src/serve_snow_dashboard.py").path) { return adjacent }
        return URL(fileURLWithPath: Bundle.main.object(forInfoDictionaryKey: "SnowlineProjectRoot") as? String ?? adjacent.path)
    }

    var pythonExecutable: URL {
        let local = projectRoot.appendingPathComponent(".venv/bin/python")
        if FileManager.default.isExecutableFile(atPath: local.path) { return local }
        if let configured = Bundle.main.object(forInfoDictionaryKey: "SnowlinePython") as? String,
           FileManager.default.isExecutableFile(atPath: configured) {
            return URL(fileURLWithPath: configured)
        }
        return URL(fileURLWithPath: "/usr/bin/env")
    }

    func request(_ path: String, body: [String: Any]? = nil) async throws -> [String: Any] {
        var request = URLRequest(url: URL(string: baseURL + path)!)
        request.timeoutInterval = path == "/api/health" ? 2 : 30
        request.cachePolicy = .reloadIgnoringLocalCacheData
        if let body = body {
            request.httpMethod = "POST"
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
            request.httpBody = try JSONSerialization.data(withJSONObject: body)
        }
        let (data, response) = try await URLSession.shared.data(for: request)
        guard let response = response as? HTTPURLResponse, (200..<300).contains(response.statusCode),
              let result = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "Snowline", code: 1, userInfo: [NSLocalizedDescriptionKey: "The local forecast service returned an unexpected response."])
        }
        return result
    }

    private func healthy() async -> Bool {
        guard let result = try? await request("/api/health") else { return false }
        return result["status"] as? String == "ok" && result["local_only"] as? Bool == true
    }

    func ensureServer(activate: Bool = true) async throws {
        if await healthy() { return }
        if let startup = startup { return try await startup.value }
        let task = Task { @MainActor in
            let script = self.projectRoot.appendingPathComponent("src/serve_snow_dashboard.py")
            guard FileManager.default.fileExists(atPath: script.path) else {
                throw NSError(domain: "Snowline", code: 2, userInfo: [NSLocalizedDescriptionKey: "Keep the widget in the Snowline project's desktop folder, then reopen it."])
            }
            let process: Process
            if let existing = self.serverProcess, existing.isRunning {
                process = existing
            } else {
                process = Process()
                let python = self.pythonExecutable
                process.executableURL = python
                process.arguments = (python.path == "/usr/bin/env" ? ["python3"] : []) + [script.path, "--port", "8765"]
                process.currentDirectoryURL = self.projectRoot
                let logURL = self.projectRoot.appendingPathComponent("outputs/dashboard/widget-server.log")
                try FileManager.default.createDirectory(at: logURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                if !FileManager.default.fileExists(atPath: logURL.path) { FileManager.default.createFile(atPath: logURL.path, contents: nil) }
                let log = try FileHandle(forWritingTo: logURL)
                try log.seekToEnd()
                process.standardOutput = log
                process.standardError = log
                if activate { NSApp.activate(ignoringOtherApps: true) }
                try process.run()
                try? log.close()
                self.serverProcess = process
            }
            self.message = "Starting local service…"
            for _ in 0..<60 {
                try await Task.sleep(nanoseconds: 1_000_000_000)
                if await self.healthy() { return }
                if !process.isRunning { break }
            }
            throw NSError(domain: "Snowline", code: 3, userInfo: [NSLocalizedDescriptionKey: "Service not ready. Check the project setup and outputs/dashboard/widget-server.log, then try Refresh all again."])
        }
        startup = task
        defer { startup = nil }
        try await task.value
    }

    func load() async {
        let id = UUID()
        requestID = id
        let selected = site
        loading = true
        failed = false
        message = "Reading saved forecast…"
        payload = nil
        do {
            try await ensureServer()
            let result = try await request("/api/forecast?site=" + selected)
            guard id == requestID else { return }
            payload = result
            message = ""
        } catch {
            guard id == requestID else { return }
            failed = true
            message = error.localizedDescription
        }
        if id == requestID { loading = false }
    }

    func refresh(automatic: Bool = false) async {
        guard !refreshing else { return }
        if automatic && !autoUpdater.mayStartRefresh { return }
        refreshing = true
        failed = false
        message = "Updating radar and weather for all four mountains…"
        let selected = site
        defer { refreshing = false }
        do {
            try await ensureServer(activate: !automatic)
            if automatic && !autoUpdater.mayStartRefresh {
                message = "Auto-update skipped. Wi-Fi is unavailable or Auto-update was turned off."
                return
            }
            let job = try await request("/api/refresh-all", body: [:])
            guard let jobID = job["job_id"] as? String else { throw URLError(.badServerResponse) }
            for _ in 0..<600 {
                try await Task.sleep(nanoseconds: 2_000_000_000)
                let state = try await request("/api/jobs/" + jobID)
                let status = state["status"] as? String ?? ""
                message = state["message"] as? String ?? "Updating all four mountains…"
                if status == "succeeded" || status == "partial" || status == "failed" {
                    // Fetch the selected mountain after the one shared batch.
                    // Never retimestamp a retained forecast when a source fails.
                    payload = try await request("/api/forecast?site=" + selected)
                    let result = state["result"] as? [String: Any]
                    failed = status == "failed" || status == "partial" || result?["status"] as? String == "partial"
                    if status == "failed" && message.isEmpty { message = "Refresh incomplete. Saved forecasts keep their original times." }
                    return
                }
            }
            message = "The update is still running. Open Full forecast for status."
        } catch {
            failed = true
            message = "Refresh unavailable. " + error.localizedDescription
        }
    }

    var selectedRows: [[String: Any]] {
        let all = payload?["hourly"] as? [[String: Any]] ?? []
        let boundary = Date(timeIntervalSince1970: ceil(Date().timeIntervalSince1970 / 3600) * 3600)
        return all.filter { row in
            guard let start = parseDate(row["start"] as? String) else { return false }
            return start >= boundary && start < boundary.addingTimeInterval(Double(hours) * 3600)
        }.sorted { ($0["start"] as? String ?? "") < ($1["start"] as? String ?? "") }
    }
    var amount: Double? {
        let rows = selectedRows
        let values = rows.compactMap { $0["snowfall_inches"] as? Double }
        guard rows.count == hours && values.count == hours else { return nil }
        return values.reduce(0, +)
    }
    var unit: String { metric ? "cm" : "in" }
    var totalText: String {
        guard let amount = amount else { return "—" }
        let converted = amount * (metric ? 2.54 : 1)
        return converted > 0 && converted < 0.01 ? "<0.01" : String(format: "%.2f", converted)
    }
    var siteInfo: [String: Any] { payload?["site"] as? [String: Any] ?? [:] }
    func dateText(_ date: Date, format: String) -> String {
        let formatter = DateFormatter()
        formatter.dateFormat = format
        formatter.timeZone = TimeZone(identifier: siteInfo["timezone"] as? String ?? (site == "alta" || site == "vail" ? "America/Denver" : "America/New_York"))
        return formatter.string(from: date)
    }
    var windowText: String {
        let start = Date(timeIntervalSince1970: ceil(Date().timeIntervalSince1970 / 3600) * 3600)
        return dateText(start, format: "MMM d, h a") + " – " + dateText(start.addingTimeInterval(Double(hours)*3600), format: "MMM d, h a z")
    }
    var sourceText: String {
        let source = payload?["source"] as? [String: Any] ?? [:]
        guard let date = parseDate(source["retrieved_at"] as? String) else { return "No saved forecast yet · Refresh to fetch data" }
        let stale = source["stale"] as? Bool == true || Date().timeIntervalSince(date) > 6*3600
        return (stale ? "Older data · " : "Fetched ") + dateText(date, format: "MMM d, h:mm a z")
    }
    var targetText: String {
        if site == "jay" { return "NOAA station near Jay · below resort elevation" }
        if site == "mansfield" { return "Mount Mansfield station · not a resort-wide total" }
        if site == "vail" { return "Vail station location · mountain conditions vary" }
        return "Alta–Collins location · mountain conditions vary"
    }
    var shortMode: Bool { site == "alta" && hours <= 3 }
    var radar: [String: Any] { payload?["radar"] as? [String: Any] ?? [:] }
    var comparisonRows: [[String: Any]] {
        guard shortMode, radar["expired"] as? Bool != true, radar["stale"] as? Bool != true,
              let issue = parseDate(radar["issued_at"] as? String), Date().timeIntervalSince(issue) <= 1800,
              let start = parseDate(radar["start"] as? String), let end = parseDate(radar["end"] as? String) else { return [] }
        let rows = selectedRows.filter {
            guard let a = parseDate($0["start"] as? String), let b = parseDate($0["time"] as? String),
                  $0["precipitation_mm"] is Double else { return false }
            return a >= start && b <= end && abs(b.timeIntervalSince(a)-3600) < 1
        }
        guard zip(rows, rows.dropFirst()).allSatisfy({ previous, next in
            guard let a = parseDate(previous["time"] as? String), let b = parseDate(next["start"] as? String) else { return false }
            return abs(a.timeIntervalSince(b)) < 1
        }) else { return [] }
        return rows
    }
    var comparisonValues: [(name: String, value: Double, color: Color)] {
        let rows = comparisonRows
        guard let start = parseDate(rows.first?["start"] as? String), let end = parseDate(rows.last?["time"] as? String) else { return [] }
        var result: [(name: String, value: Double, color: Color)] = [("Open-Meteo", rows.compactMap { $0["precipitation_mm"] as? Double }.reduce(0,+), teal)]
        for model in radar["models"] as? [[String: Any]] ?? [] {
            guard model["id"] as? String == "dgmr", let points = model["hours"] as? [[String: Any]] else { continue }
            func cumulative(_ date: Date) -> Double? {
                points.first { point in
                    guard let time = parseDate(point["time"] as? String) else { return false }
                    return abs(time.timeIntervalSince(date)) < 1
                }?["mean_liquid_mm"] as? Double
            }
            guard let a = cumulative(start), let b = cumulative(end), b >= a else { continue }
            result.append(("Our model", b-a, Color.orange))
        }
        return result.count > 1 ? result : []
    }
    var comparisonWindow: String {
        guard let start = parseDate(comparisonRows.first?["start"] as? String), let end = parseDate(comparisonRows.last?["time"] as? String) else { return "" }
        return dateText(start, format: "MMM d, ha") + " – " + dateText(end, format: "MMM d, ha z")
    }
    var neural48: [String: Any] { payload?["neural48"] as? [String: Any] ?? [:] }
    var neuralTotal: [String: Any]? {
        guard neural48["status"] as? String != "unavailable" else { return nil }
        return (neural48["totals"] as? [[String: Any]] ?? []).first { $0["hours"] as? Int == hours }
    }
    var neuralSaved: Bool {
        neural48["expired"] as? Bool == true || neural48["window_started"] as? Bool == true || neural48["stale"] as? Bool == true
    }
    var neuralWindow: String {
        guard let start = parseDate(neural48["start"] as? String), let end = parseDate(neuralTotal?["end"] as? String) else { return "" }
        return dateText(start, format: "MMM d, ha") + " – " + dateText(end, format: "MMM d, ha z")
    }
    func snowText(_ value: Double?) -> String {
        guard let value = value else { return "—" }
        let converted = value * (metric ? 2.54 : 1)
        return converted > 0 && converted < 0.01 ? "<0.01" : String(format: "%.2f", converted)
    }
    var neuralBars: [(name: String, value: Double?, color: Color)] {
        guard let total = neuralTotal else { return [] }
        return [("Open-Meteo", total["provider_inches"] as? Double, teal),
                ("Our model", total["mean_inches"] as? Double, Color.orange)]
    }
    var deployed: [String: Any] { payload?["deployed"] as? [String: Any] ?? [:] }
    var deployedTotal: [String: Any]? {
        (deployed["horizons"] as? [[String: Any]] ?? []).first { $0["hours"] as? Int == hours }
    }
    var deployedLiquid: Bool { deployedTotal?["quantity"] as? String == "liquid_precipitation" }
    var deployedUnit: String { deployedLiquid ? (metric ? "mm water" : "in water") : unit }
    var deployedWindow: String {
        guard let start = parseDate(deployedTotal?["start"] as? String),
              let end = parseDate(deployedTotal?["end"] as? String) else { return "Forecast unavailable" }
        return dateText(start, format: "MMM d, ha") + " – " + dateText(end, format: "MMM d, ha z")
    }
    var deployedSaved: Bool {
        guard let total = deployedTotal else { return false }
        if total["stale"] as? Bool == true || total["window_started"] as? Bool == true || total["expired"] as? Bool == true { return true }
        return parseDate(total["start"] as? String).map { $0 < Date() } ?? false
    }
    private func validAmount(_ value: Any?) -> Double? {
        guard let value = value as? Double, value.isFinite, value >= 0 else { return nil }
        return value
    }
    var deployedBars: [(name: String, value: Double?, color: Color)] {
        guard let total = deployedTotal, total["status"] as? String != "unavailable",
              let mean = validAmount(total["mean"]) else { return [] }
        return [("Open-Meteo", validAmount(total["provider_mean"]), teal), ("Our model", mean, Color.orange)]
    }
    func deployedText(_ value: Double?) -> String {
        guard let value = value, value.isFinite, value >= 0 else { return "—" }
        let converted = deployedLiquid ? value / (metric ? 1 : 25.4) : value * (metric ? 2.54 : 1)
        let cutoff = deployedLiquid ? 0.001 : 0.01
        if converted > 0 && converted < cutoff { return deployedLiquid ? "<.001" : "<0.01" }
        return String(format: deployedLiquid ? "%.3f" : "%.2f", converted)
    }
    var deployedRange: String? {
        guard let total = deployedTotal,
              let lower = validAmount(total["lower"]), let upper = validAmount(total["upper"]), lower <= upper,
              let label = total["interval_label"] as? String, !label.isEmpty else { return nil }
        return label + ": " + deployedText(lower) + "–" + deployedText(upper) + " " + deployedUnit
    }
    func openDashboard() {
        Task {
            do {
                try await ensureServer()
                let url = URL(string: baseURL + "/?mountain=\(site)&hours=\(hours)&units=\(metric ? "metric" : "us")")!
                NSWorkspace.shared.open(url)
            } catch { failed = true; message = error.localizedDescription }
        }
    }
}

private struct SnowBars: View {
    @ObservedObject var store: ForecastStore
    var body: some View {
        Canvas { context, size in
            let baseline = size.height - 12
            var zero = Path()
            zero.move(to: CGPoint(x: 0, y: baseline)); zero.addLine(to: CGPoint(x: size.width, y: baseline))
            context.stroke(zero, with: .color(teal.opacity(0.35)), lineWidth: 1)
            let rows = store.selectedRows
            let scale = max(0.1, rows.compactMap { $0["snowfall_inches"] as? Double }.max() ?? 0)
            let step = size.width / Double(store.hours)
            let boundary = ceil(Date().timeIntervalSince1970 / 3600) * 3600
            for row in rows {
                guard let value = row["snowfall_inches"] as? Double,
                      let start = parseDate(row["start"] as? String) else { continue }
                let index = (start.timeIntervalSince1970 - boundary) / 3600
                let x = index * step
                if value == 0 {
                    var path = Path()
                    path.move(to: CGPoint(x: x, y: baseline)); path.addLine(to: CGPoint(x: x + step, y: baseline))
                    context.stroke(path, with: .color(teal), lineWidth: 2.5)
                } else {
                    let height = max(2, value / scale * (baseline - 8))
                    let rect = CGRect(x: x + step * 0.12, y: baseline-height, width: max(2, step * 0.76), height: height)
                    context.fill(Path(roundedRect: rect, cornerRadius: 1.5), with: .color(snowColor(value)))
                }
            }
        }
        .frame(height: 66)
        .accessibilityLabel("Hourly snowfall chart. " + (store.amount == 0 ? "No snowfall forecast, shown at zero." : "Total " + store.totalText + " " + store.unit))
    }
}

private struct ForecastPanel: View {
    @ObservedObject var store: ForecastStore
    var close: () -> Void
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack {
                Image(systemName: "mountain.2.fill").font(.system(size: 20)).foregroundColor(teal)
                Text("SNOWLINE").font(.system(size: 11, weight: .bold)).tracking(2)
                Spacer()
                Button(action: close) { Image(systemName: "xmark").font(.system(size: 12, weight: .semibold)).frame(width: 24, height: 24) }
                    .buttonStyle(.plain).help("Collapse to floating icon").accessibilityLabel("Collapse widget")
            }
            HStack {
                Picker("Mountain", selection: $store.site) {
                    ForEach(mountains, id: \.0) { mountain in Text(mountain.1 + " · " + mountain.2).tag(mountain.0) }
                }.labelsHidden().frame(maxWidth: .infinity).disabled(store.refreshing)
                    .onChange(of: store.site) { _ in Task { await store.load() } }
                Button(store.metric ? "cm · °C" : "in · °F") { store.metric.toggle() }
                    .buttonStyle(.bordered).controlSize(.small).help("Switch US / Metric units")
            }
            HStack(spacing: 5) {
                ForEach(horizons, id: \.self) { hours in
                    Button { store.hours = hours } label: {
                        Text("\(hours)h").font(.system(size: 12, weight: store.hours == hours ? .semibold : .regular))
                            .frame(maxWidth: .infinity).padding(.vertical, 8)
                            .background(store.hours == hours ? teal : Color.white)
                            .foregroundColor(store.hours == hours ? .white : ink.opacity(0.7))
                            .clipShape(RoundedRectangle(cornerRadius: 7))
                    }.buttonStyle(.plain).accessibilityLabel("Next \(hours) hours").accessibilityAddTraits(store.hours == hours ? .isSelected : [])
                }
            }
            ScrollView(.vertical) {
            VStack(alignment: .leading, spacing: 14) {
            DeployedComparison(store: store)
            VStack(alignment: .leading, spacing: 5) {
                HStack(alignment: .firstTextBaseline, spacing: 7) {
                    Text(store.totalText).font(.system(size: 42, weight: .semibold, design: .rounded)).monospacedDigit()
                    Text(store.unit + " snow").font(.system(size: 15)).foregroundColor(ink.opacity(0.7))
                    Spacer()
                    if store.loading || store.refreshing { ProgressView().controlSize(.small).frame(width: 20) }
                }
                Text("Next \(store.hours) full hours").font(.system(size: 13, weight: .medium))
                Text(store.windowText).font(.system(size: 10)).foregroundColor(ink.opacity(0.65))
                SnowBars(store: store)
                HStack {
                    Text("0")
                    Spacer()
                    Text("Hourly intensity")
                    LinearGradient(colors: [snowColor(0), snowColor(0.5), snowColor(1)], startPoint: .leading, endPoint: .trailing).frame(width: 46, height: 5).clipShape(Capsule())
                    Text(store.metric ? "2.54+ cm/h" : "1+ in/h")
                }.font(.system(size: 9)).foregroundColor(ink.opacity(0.6))
                Text(store.amount == nil && !store.loading ? "Incomplete forecast · \(store.selectedRows.filter { $0["snowfall_inches"] is Double }.count)/\(store.hours) hours available" : " ")
                    .font(.system(size: 10)).foregroundColor(.orange).frame(height: 12)
            }.padding(14).background(Color.white).clipShape(RoundedRectangle(cornerRadius: 12))
            VStack(alignment: .leading, spacing: 5) {
                Text("Open-Meteo · external hourly forecast").font(.system(size: 11, weight: .semibold)).foregroundColor(teal)
                Text(store.sourceText).font(.system(size: 10))
            }
            }
            }.frame(height: 282)
            Text(store.message.isEmpty ? " " : store.message).font(.system(size: 10)).foregroundColor(store.failed ? .red : ink.opacity(0.7)).lineLimit(3).frame(height: 36, alignment: .topLeading)
            AutoUpdateControl(updater: store.autoUpdater)
            HStack(spacing: 8) {
                Button { Task { await store.refresh() } } label: {
                    Label(store.refreshing ? "Updating…" : "Refresh all", systemImage: "arrow.clockwise").frame(height: 23)
                }.buttonStyle(.bordered).disabled(store.refreshing || store.loading)
                Button { store.openDashboard() } label: {
                    Label("Full forecast", systemImage: "arrow.up.right.square").frame(maxWidth: .infinity).frame(height: 23)
                }.buttonStyle(.borderedProminent).tint(teal)
            }
            HStack {
                Text("Model errors & sources in the full dashboard.").font(.system(size: 9)).foregroundColor(ink.opacity(0.55))
                Spacer()
                Button("Quit") { NSApplication.shared.terminate(nil) }.buttonStyle(.plain).font(.system(size: 10)).foregroundColor(ink.opacity(0.65))
            }
        }
        .padding(20).frame(width: 360).foregroundColor(ink)
        .background(Color(red: 0.94, green: 0.96, blue: 0.94))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .overlay(RoundedRectangle(cornerRadius: 20).stroke(teal.opacity(0.15), lineWidth: 1))
        .environment(\.colorScheme, .light)
    }
}

private struct AutoUpdateControl: View {
    @ObservedObject var updater: HourlyAutoUpdater
    var body: some View {
        HStack(spacing: 8) {
            Toggle("Auto-update", isOn: $updater.enabled)
                .toggleStyle(.switch).controlSize(.mini)
                .font(.system(size: 11)).fixedSize()
                .help("Refresh all four mountains at HH:00:00 on this Mac's clock, while Snowline is open and Wi-Fi is available. Sleeping or missed hours are skipped.")
            Spacer(minLength: 0)
            Text(updater.statusText).font(.system(size: 9)).foregroundColor(ink.opacity(0.65)).lineLimit(1)
        }
    }
}

// One comparison layout keeps method order, colors and bar scales consistent
// across short radar windows and full-day snowfall windows.
private struct ForecastComparisonCard: View {
    let title: String
    let values: [(name: String, value: Double?, color: Color)]
    let format: (Double?) -> String
    let emptyMessage: String
    var detail: String? = nil
    var saved: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(title).font(.system(size: 11, weight: .semibold)).fixedSize(horizontal: false, vertical: true)
                Spacer()
                if saved { Text("Saved").font(.system(size: 9, weight: .medium)).foregroundColor(.orange) }
            }
            if values.isEmpty {
                Text(emptyMessage).font(.system(size: 11)).foregroundColor(ink.opacity(0.7))
                    .fixedSize(horizontal: false, vertical: true)
            } else {
                ForEach(values, id: \.name) { row in
                    HStack(spacing: 5) {
                        Text(row.name).font(.system(size: 10)).frame(width: 89, alignment: .leading)
                            .lineLimit(2).fixedSize(horizontal: false, vertical: true)
                        GeometryReader { geo in
                            if let value = row.value {
                                let peak = max(0.01, values.compactMap { $0.value }.max() ?? 0.01)
                                RoundedRectangle(cornerRadius: 2).fill(row.color)
                                    .frame(width: value == 0 ? 1 : max(2, geo.size.width * value / peak), height: 10)
                            }
                        }.frame(height: 10)
                        Text(format(row.value)).font(.system(size: 10, design: .monospaced))
                            .frame(width: 40, alignment: .trailing)
                    }
                }
                if let detail = detail {
                    Text(detail).font(.system(size: 10)).foregroundColor(ink.opacity(0.7))
                }
            }
        }.padding(12).background(Color.white).clipShape(RoundedRectangle(cornerRadius: 12))
    }
}

private struct DeployedComparison: View {
    @ObservedObject var store: ForecastStore
    var body: some View {
        ForecastComparisonCard(
            title: store.deployedWindow + " · " + store.deployedUnit,
            values: store.deployedBars,
            format: { store.deployedText($0) },
            emptyMessage: "Refresh all to load Our model. Full forecast has source and availability details.",
            detail: store.deployedRange,
            saved: store.deployedSaved)
    }
}

private struct NeuralComparison: View {
    @ObservedObject var store: ForecastStore
    var body: some View {
        ForecastComparisonCard(
            title: (store.neuralWindow.isEmpty ? "Forecast unavailable" : store.neuralWindow) + " · " + store.unit,
            values: store.neuralBars,
            format: { store.snowText($0) },
            emptyMessage: "Refresh all to check the 48-hour model. Full forecast has source and availability details.",
            detail: store.neuralTotal.map { total in
                "80% model range: " + store.snowText(total["q10_inches"] as? Double) + "–" + store.snowText(total["q90_inches"] as? Double) + " " + store.unit
            },
            saved: store.neuralSaved)
    }
}

private struct ShortComparison: View {
    @ObservedObject var store: ForecastStore
    var body: some View {
        ForecastComparisonCard(
            title: (store.comparisonWindow.isEmpty ? "Forecast unavailable" : store.comparisonWindow) + " · " + (store.metric ? "mm water" : "in water"),
            values: store.comparisonValues.map { ($0.name, Optional($0.value), $0.color) },
            format: { value in
                guard let value = value else { return "—" }
                let number = value / (store.metric ? 1 : 25.4)
                return number > 0 && number < 0.001 ? "<.001" : String(format: "%.3f", number)
            },
            emptyMessage: "Refresh all to load the model forecast.")
    }
}

private final class FloatingPanel: NSPanel {
    override var canBecomeKey: Bool { true }
    override var canBecomeMain: Bool { false }
    var onEscape: (() -> Void)?
    override func cancelOperation(_ sender: Any?) { onEscape?() }
}

private final class MountainButton: NSButton {
    var clicked: (() -> Void)?
    var moved: (() -> Void)?
    override var acceptsFirstResponder: Bool { true }
    override func draw(_ dirtyRect: NSRect) {
        NSGraphicsContext.saveGraphicsState()
        defer { NSGraphicsContext.restoreGraphicsState() }
        if isFlipped {
            let transform = NSAffineTransform()
            transform.translateX(by: 0, yBy: bounds.height)
            transform.scaleX(by: 1, yBy: -1)
            transform.concat()
        }
        let rect = bounds.insetBy(dx: 5, dy: 5)
        let background = NSBezierPath(roundedRect: rect, xRadius: 22, yRadius: 22)
        NSColor(red: 0.07, green: 0.35, blue: 0.38, alpha: 0.98).setFill(); background.fill()
        NSColor.white.withAlphaComponent(0.22).setStroke(); background.lineWidth = 1; background.stroke()
        let mountain = NSBezierPath()
        mountain.move(to: NSPoint(x: 16, y: 23)); mountain.line(to: NSPoint(x: 31, y: 47))
        mountain.line(to: NSPoint(x: 40, y: 33)); mountain.line(to: NSPoint(x: 47, y: 42))
        mountain.line(to: NSPoint(x: 59, y: 23)); mountain.close()
        NSColor(red: 0.68, green: 0.85, blue: 0.84, alpha: 1).setFill(); mountain.fill()
        let cap = NSBezierPath()
        cap.move(to: NSPoint(x: 25, y: 37)); cap.line(to: NSPoint(x: 31, y: 47))
        cap.line(to: NSPoint(x: 37, y: 37)); cap.line(to: NSPoint(x: 32, y: 39)); cap.line(to: NSPoint(x: 29, y: 36)); cap.close()
        NSColor.white.setFill(); cap.fill()
        NSColor.white.withAlphaComponent(0.9).setFill()
        NSBezierPath(ovalIn: NSRect(x: 48, y: 50, width: 4, height: 4)).fill()
    }
    override func mouseDown(with event: NSEvent) {
        guard let window = window else { return }
        let origin = window.frame.origin
        let start = window.convertPoint(toScreen: event.locationInWindow)
        var dragged = false
        while let next = window.nextEvent(matching: [.leftMouseDragged, .leftMouseUp]) {
            let now = window.convertPoint(toScreen: next.locationInWindow)
            let dx = now.x - start.x, dy = now.y - start.y
            if hypot(dx, dy) > 4 { dragged = true }
            if dragged { window.setFrameOrigin(NSPoint(x: origin.x + dx, y: origin.y + dy)); moved?() }
            if next.type == .leftMouseUp { break }
        }
        if dragged { moved?() } else { clicked?() }
    }
    override func keyDown(with event: NSEvent) {
        if event.keyCode == 49 || event.keyCode == 36 { clicked?() } else { super.keyDown(with: event) }
    }
    override func accessibilityPerformPress() -> Bool { clicked?(); return true }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    let store = ForecastStore()
    private var bubble: FloatingPanel!
    private var drawer: FloatingPanel!
    private var statusItem: NSStatusItem!
    private var screenObserver: NSObjectProtocol?
    private var clockTimer: Timer?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        bubble = makePanel(size: NSSize(width: 74, height: 74))
        bubble.title = "Snowline floating widget"
        let button = MountainButton(frame: NSRect(x: 0, y: 0, width: 74, height: 74))
        button.isBordered = false
        button.title = ""
        button.setAccessibilityLabel("Snowline. Click for forecast; drag to move.")
        button.toolTip = "Snowline · Click for forecast · Drag to move"
        button.clicked = { [weak self] in self?.toggle() }
        button.moved = { [weak self] in self?.positionChanged() }
        let menu = NSMenu()
        menu.addItem(withTitle: "Open full forecast", action: #selector(openDashboard), keyEquivalent: "")
        menu.addItem(withTitle: "Quit Snowline", action: #selector(quit), keyEquivalent: "q")
        for item in menu.items { item.target = self }
        button.menu = menu
        bubble.contentView = button
        let screen = NSScreen.main?.visibleFrame ?? NSRect(x: 0, y: 0, width: 1440, height: 900)
        let saved = UserDefaults.standard.string(forKey: "position").map(NSPointFromString)
        bubble.setFrameOrigin(saved ?? NSPoint(x: screen.maxX - 100, y: screen.minY + 100))
        clampBubble()
        bubble.orderFrontRegardless()

        drawer = makePanel(size: NSSize(width: 360, height: 510))
        drawer.title = "Snowline forecast"
        drawer.onEscape = { [weak self] in self?.drawer.orderOut(nil) }
        let hosting = NSHostingView(rootView: ForecastPanel(store: store, close: { [weak self] in self?.drawer.orderOut(nil) }))
        drawer.contentView = hosting
        // Fixed room for status/availability notices; the content sizes itself vertically.
        let size = hosting.fittingSize
        drawer.setContentSize(NSSize(width: 360, height: max(size.height, 490)))

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        statusItem.button?.image = NSImage(systemSymbolName: "cloud.snow.fill", accessibilityDescription: "Snowline")
        statusItem.button?.toolTip = "Snowline forecast widget"
        let statusMenu = NSMenu()
        statusMenu.addItem(withTitle: "Show Snowline", action: #selector(showWidget), keyEquivalent: "")
        statusMenu.addItem(withTitle: "Open full forecast", action: #selector(openDashboard), keyEquivalent: "")
        statusMenu.addItem(NSMenuItem.separator())
        statusMenu.addItem(withTitle: "Quit Snowline", action: #selector(quit), keyEquivalent: "q")
        for item in statusMenu.items { item.target = self }
        statusItem.menu = statusMenu
        screenObserver = NotificationCenter.default.addObserver(forName: NSApplication.didChangeScreenParametersNotification, object: nil, queue: .main) { [weak self] _ in
            Task { @MainActor in self?.clampBubble(); self?.positionDrawer() }
        }
        clockTimer = Timer.scheduledTimer(withTimeInterval: 60, repeats: true) { [weak self] _ in
            Task { @MainActor in self?.store.objectWillChange.send() }
        }
        Task { await store.load() }
    }
    private func makePanel(size: NSSize) -> FloatingPanel {
        let panel = FloatingPanel(contentRect: NSRect(origin: .zero, size: size), styleMask: [.borderless, .nonactivatingPanel], backing: .buffered, defer: false)
        panel.level = .floating
        panel.isOpaque = false
        panel.backgroundColor = .clear
        panel.hasShadow = true
        panel.hidesOnDeactivate = false
        panel.isReleasedWhenClosed = false
        panel.collectionBehavior = [.canJoinAllSpaces, .fullScreenAuxiliary]
        return panel
    }
    private func clampBubble() {
        let frame = bubble.frame
        let screen = NSScreen.screens.first(where: { $0.visibleFrame.intersects(frame) })?.visibleFrame ?? NSScreen.main!.visibleFrame
        bubble.setFrameOrigin(NSPoint(x: min(max(frame.minX, screen.minX + 8), screen.maxX - frame.width - 8),
                                      y: min(max(frame.minY, screen.minY + 8), screen.maxY - frame.height - 8)))
    }
    private func positionDrawer() {
        guard drawer != nil else { return }
        let frame = bubble.frame
        let screen = NSScreen.screens.first(where: { $0.visibleFrame.contains(NSPoint(x: frame.midX, y: frame.midY)) })?.visibleFrame ?? NSScreen.main!.visibleFrame
        let x = frame.minX - drawer.frame.width - 10 >= screen.minX ? frame.minX - drawer.frame.width - 10 : frame.maxX + 10
        drawer.setFrameOrigin(NSPoint(x: min(max(x, screen.minX + 8), screen.maxX - drawer.frame.width - 8),
                                      y: min(max(frame.minY, screen.minY + 8), screen.maxY - drawer.frame.height - 8)))
    }
    private func positionChanged() {
        clampBubble()
        UserDefaults.standard.set(NSStringFromPoint(bubble.frame.origin), forKey: "position")
        if drawer?.isVisible == true { positionDrawer() }
    }
    private func toggle() {
        if drawer.isVisible { drawer.orderOut(nil) } else { showWidget() }
    }
    @objc func showWidget() {
        clampBubble()
        bubble.orderFrontRegardless()
        positionDrawer()
        drawer.makeKeyAndOrderFront(nil)
    }
    @objc func openDashboard() { store.openDashboard() }
    @objc func quit() { NSApp.terminate(nil) }
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool { showWidget(); return true }
}

@main
struct SnowlineApp {
    @MainActor static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.run()
    }
}
