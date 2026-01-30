"use client";

import ChatUI from "@/components/playground/chat_ui/ChatUI";
import CompareUI from "@/components/playground/compareUI/CompareUI";
import { TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

export default function PlaygroundPage() {
  const { accessToken, userRole, userId, disabledPersonalKeyCreation, token } = useAuthorized();

  return (
    <TabGroup className="h-full w-full">
      <TabList className="mb-0">
        <Tab>Chat</Tab>
        <Tab>Compare</Tab>
      </TabList>
      <TabPanels className="h-full">
        <TabPanel className="h-full">
          <ChatUI
            accessToken={accessToken}
            token={token}
            userRole={userRole}
            userID={userId}
            disabledPersonalKeyCreation={disabledPersonalKeyCreation}
          />
        </TabPanel>
        <TabPanel className="h-full">
          <CompareUI accessToken={accessToken} disabledPersonalKeyCreation={disabledPersonalKeyCreation} />
        </TabPanel>
      </TabPanels>
    </TabGroup>
  );
}
