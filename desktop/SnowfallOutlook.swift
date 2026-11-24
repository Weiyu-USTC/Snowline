import Foundation

func parseDate(_ text: String?) -> Date? {
    guard let text = text else { return nil }
    let formatter = ISO8601DateFormatter()
    formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    if let date = formatter.date(from: text) { return date }
    formatter.formatOptions = [.withInternetDateTime]
    return formatter.date(from: text)
}

struct SnowfallOutlook {
    let start: Date
    let end: Date
    let amounts: [Double?] // Inches at each successive hour in the chosen window
    let paired: Bool
    let source: [String: Any]

    var total: Double? {
        let valid = amounts.compactMap { $0 }
        return valid.count == amounts.count ? valid.reduce(0, +) : nil
    }

    init(payload: [String: Any], record: [String: Any]?, hours: Int, now: Date) {
        let rollingStart = Date(timeIntervalSince1970: ceil(now.timeIntervalSince1970 / 3600) * 3600)
        let duration = TimeInterval(hours * 3600)
        let rows: [[String: Any]]
        if let record = record, record["quantity"] as? String == "snowfall",
           record["status"] as? String != "unavailable",
           let model = record["mean"] as? Double, model.isFinite, model >= 0,
           record.keys.contains("provider_hourly"),
           let first = parseDate(record["start"] as? String), let last = parseDate(record["end"] as? String),
           last.timeIntervalSince(first) == duration {
            start = first
            end = last
            paired = true
            source = (record["sources"] as? [String: Any])?["provider"] as? [String: Any] ?? [:]
            if let provider = record["provider_mean"] as? Double, provider.isFinite, provider >= 0 {
                rows = record["provider_hourly"] as? [[String: Any]] ?? []
            } else {
                rows = []
            }
        } else {
            start = rollingStart
            end = rollingStart.addingTimeInterval(duration)
            paired = false
            source = payload["source"] as? [String: Any] ?? [:]
            rows = payload["hourly"] as? [[String: Any]] ?? []
        }

        var slots = [Double?](repeating: nil, count: hours)
        var seen = Set<Int>()
        for row in rows {
            guard let time = parseDate(row["start"] as? String),
                  let stop = parseDate(row["time"] as? String),
                  stop.timeIntervalSince(time) == 3600 else { continue }
            let offset = time.timeIntervalSince(start) / 3600
            guard offset >= 0, offset < Double(hours), offset == offset.rounded() else { continue }
            let slot = Int(offset)
            if seen.insert(slot).inserted, let value = row["snowfall_inches"] as? Double,
               value.isFinite, value >= 0 {
                slots[slot] = value
            } else {
                slots[slot] = nil
            }
        }
        amounts = slots
    }
}
