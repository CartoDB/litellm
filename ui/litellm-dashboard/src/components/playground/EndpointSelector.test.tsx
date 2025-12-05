import { render, waitFor } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import EndpointSelector from "./chat_ui/EndpointSelector";
import { ENDPOINT_OPTIONS } from "./chat_ui/chatConstants";

describe("EndpointSelector", () => {
  Object.values(ENDPOINT_OPTIONS).forEach((endpointType) => {
    it(`should render the endpoint selector for ${endpointType.value}`, async () => {
      const { getByText } = render(<EndpointSelector endpointType={endpointType.value} onEndpointChange={() => {}} />);
      await waitFor(() => {
        expect(getByText(endpointType.label)).toBeInTheDocument();
      });
    });
  });
});
