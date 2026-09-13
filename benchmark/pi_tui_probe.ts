import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Test-only lifecycle marker: pexpect waits for the real TUI agent turn to
// finish before sending /exit. This does not alter prompts, tools, or results.
export default function (pi: ExtensionAPI) {
	pi.on("agent_end", async () => {
		process.stdout.write("\n__PEAKSMCP_PI_AGENT_END__\n");
	});
}
